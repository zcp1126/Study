# 训练入口：负责组织数据加载、模型创建、训练、验证及日志记录。
import sys
import torch

from timm.utils import AverageMeter, accuracy, NativeScaler
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from tqdm import tqdm

# 从 setup.py 读取已合并 YAML 配置、设备设置和日志对象。
from models.build import build_models, freeze_backbone
from setup import config, log
from utils.data_loader import build_loader
from utils.eval import *
from utils.info import *
from utils.optimizer import build_optimizer
from utils.scheduler import build_scheduler

# TensorBoard 为可选依赖；不可用时训练流程仍可继续，但不会写入曲线日志。
try:
	from torch.utils.tensorboard import SummaryWriter
except:
	pass


def build_model(config, num_classes):
	"""按配置创建 MPSA 模型，加载预训练权重并记录模型规模。"""
	# build_models 会根据 config.model.type/name 选择 Swin、ResNet 或 ViT 骨干网络。
	model = build_models(config, num_classes)
	# if torch.__version__[0] == '2' and sys.platform != 'win32':
	# 	# torch.set_float32_matmul_precision('high')
	# 	model = torch.compile(model)
	# 将模型移动到 setup.py 指定的 CUDA 设备；当前配置为单卡 GPU 0。
	model.to(config.device)
	# 可选冻结骨干网络，仅训练后续 MPSA 模块。
	freeze_backbone(model, config.train.freeze_backbone)
	# 未包裹 DDP 的模型对象，用于保存权重等操作。
	model_without_ddp = model
	n_parameters = count_parameters(model)

	# yacs 配置默认冻结；临时解冻以写入运行时得到的类别数和参数量。
	config.defrost()
	config.model.num_classes = num_classes
	config.model.parameters = f'{n_parameters:.3f}M'
	config.freeze()
	if config.local_rank in [-1, 0]:
		PSetting(log, 'Model Structure', config.model.keys(), config.model.values(), rank=config.local_rank)
		log.save(model)
	return model, model_without_ddp


def main(config):
	"""执行一次完整训练：初始化资源、按 epoch 训练和验证，并保存最佳检查点。"""
	# 分别统计总耗时、初始化耗时、训练耗时和验证耗时。
	total_timer = Timer()
	prepare_timer = Timer()
	prepare_timer.start()
	train_timer = Timer()
	eval_timer = Timer()
	total_timer.start()
	# 如启用 config.write，向 output/<dataset>/<experiment>/ 写入 TensorBoard 日志。
	writer = None
	if config.write:
		try:
			writer = SummaryWriter(config.data.log_path)
		except:
			pass

	# 创建训练/测试 DataLoader，并得到数据集类别数和可选 Mixup 函数。
	train_loader, test_loader, num_classes, train_samples, test_samples, mixup_fn = build_loader(config)
	step_per_epoch = len(train_loader)
	total_batch_size = config.data.batch_size * get_world_size()
	steps = config.train.epochs * step_per_epoch

	# 创建模型；此阶段会加载 pretrained/Swin Base.pth 预训练权重。
	model, model_without_ddp = build_model(config, num_classes)

	# 多 GPU 场景才会包裹 DistributedDataParallel；单卡时 local_rank 为 -1。
	if config.local_rank != -1:
		model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank],
		                                                  broadcast_buffers=False,
		                                                  find_unused_parameters=False)
	# backbone_low_lr = config.model.type.lower() == 'resnet'
	# optimizer = build_optimizer(config, model, backbone_low_lr)
	# 优化器、混合精度梯度缩放器和按 step 更新的学习率调度器。
	optimizer = build_optimizer(config, model, False)
	loss_scaler = NativeScalerWithGradNormCount()
	scheduler = build_scheduler(config, optimizer, step_per_epoch)

	# 根据数据增强策略选择损失函数：Mixup、标签平滑或普通交叉熵。
	best_acc, best_epoch, train_accuracy = 0., 0., 0.

	if config.data.mixup > 0.:
		criterion = SoftTargetCrossEntropy()
	elif config.model.label_smooth:
		criterion = LabelSmoothingCrossEntropy(smoothing=config.model.label_smooth)
	else:
		criterion = torch.nn.CrossEntropyLoss()

	# 断点续训：恢复模型、优化器和调度器状态；eval_mode 时恢复后仅验证一次。
	if config.model.resume:
		best_acc = load_checkpoint(config, model, optimizer, scheduler, loss_scaler, log)
		best_epoch = config.train.start_epoch
		accuracy, loss = valid(config, model, test_loader, best_epoch, train_accuracy,writer,True)
		log.info(f'Epoch {best_epoch+1:^3}/{config.train.epochs:^3}: Accuracy {accuracy:2.3f}    '
		         f'BA {best_acc:2.3f}    BE {best_epoch+1:3}    '
		         f'Loss {loss:1.4f}    TA {train_accuracy * 100:2.2f}')
		if config.misc.eval_mode:
			return

	# 吞吐量模式仅测试推理速度，不执行训练。
	if config.misc.throughput:
		throughput(test_loader, model, log, config.local_rank)
		return

	# 每个 epoch 的指标会同步写入日志中的 Markdown 表格。
	mark_table = PMarkdownTable(log, ['Epoch', 'Accuracy', 'Best Accuracy',
	                                  'Best Epoch', 'Loss'], rank=config.local_rank)

	# 等待 CUDA 初始化任务结束，以得到较准确的数据与模型准备耗时。
	torch.cuda.synchronize()
	prepare_time = prepare_timer.stop()
	PSetting(log, 'Training Information',
	         ['Train samples', 'Test samples', 'Total Batch Size', 'Load Time', 'Train Steps',
	          'Warm Epochs'],
	         [train_samples, test_samples, total_batch_size,
	          f'{prepare_time:.0f}s', steps, config.train.warmup_epochs],
	         newline=2, rank=config.local_rank)

	# 按配置的起始 epoch 至总 epoch 数执行训练和验证。
	sub_title(log, 'Start Training', rank=config.local_rank)
	for epoch in range(config.train.start_epoch, config.train.epochs):
		train_timer.start()
		# 分布式训练时，每轮重设采样器随机种子，避免各进程读取相同样本顺序。
		if config.local_rank != -1:
			train_loader.sampler.set_epoch(epoch)
		# list1 = list(model.named_parameters())
		# print(list1[76])

		if not config.misc.eval_mode:
			train_accuracy = train_one_epoch(config, model, criterion, train_loader, optimizer,
			                                 epoch, scheduler, loss_scaler, mixup_fn, writer)
		train_timer.stop()

		# 按 eval_every 间隔验证；最后一个 epoch 始终验证。
		eval_timer.start()
		if (epoch + 1) % config.misc.eval_every == 0 or epoch + 1 == config.train.epochs:
			accuracy, loss = valid(config, model, test_loader, epoch, train_accuracy, writer,False)
			if config.local_rank in [-1, 0]:
				# 仅在验证精度刷新时保存最佳检查点。
				if best_acc < accuracy:
					best_acc = accuracy
					best_epoch = epoch + 1
					if config.write and epoch > 1 and config.train.checkpoint:
						save_checkpoint(config, epoch, model, best_acc, optimizer, scheduler, loss_scaler, log)
				log.info(f'Epoch {epoch + 1:^3}/{config.train.epochs:^3}: Accuracy {accuracy:2.3f}    '
				         f'BA {best_acc:2.3f}    BE {best_epoch:3}    '
				         f'Loss {loss:1.4f}    TA {train_accuracy * 100:2.2f}')
				if config.write:
					mark_table.add(log, [epoch + 1, f'{accuracy:2.3f}',
					                     f'{best_acc:2.3f}', best_epoch, f'{loss:1.5f}'], rank=config.local_rank)
			pass  # Eval
		eval_timer.stop()
		pass  # Train

	# 关闭日志并汇总训练、验证和同步开销。
	if writer is not None:
		writer.close()
	train_time = train_timer.sum / 60
	eval_time = eval_timer.sum / 60
	total_time = train_time + eval_time
	total_time_true = total_timer.stop()
	total_time_true = total_time_true/60
	PSetting(log, "Finish Training",
	         ['Best Accuracy', 'Best Epoch', 'Training Time', 'Testing Time', 'Syncthing Time','Total Time'],
	         [f'{best_acc:2.3f}', best_epoch, f'{train_time:.2f} min', f'{eval_time:.2f} min', f'{total_time_true-total_time:.2f} min' ,f'{total_time_true:.2f} min'],
	         newline=2, rank=config.local_rank)


def train_one_epoch(config, model, criterion, train_loader, optimizer, epoch, scheduler, loss_scaler, mixup_fn=None,
                    writer=None):
	"""完成一个训练 epoch，并返回该轮训练集准确率。"""
	# 启用训练模式，使 Dropout、BatchNorm 等层按训练逻辑工作。
	model.train()
	optimizer.zero_grad()

	step_per_epoch = len(train_loader)
	loss_meter = AverageMeter()
	norm_meter = AverageMeter()
	scaler_meter = AverageMeter()
	epochs = config.train.epochs

	loss1_meter = AverageMeter()
	loss2_meter = AverageMeter()
	loss3_meter = AverageMeter()

	p_bar = tqdm(total=step_per_epoch,
	             desc=f'Train {epoch + 1:^3}/{epochs:^3}',
	             dynamic_ncols=True,
	             ascii=True,
	             disable=config.local_rank not in [-1, 0])
	all_preds, all_label = None, None
	for step, (x, y) in enumerate(train_loader):
		# global_step 供学习率调度和 TensorBoard 横轴使用。
		global_step = epoch * step_per_epoch + step
		# non_blocking 配合 pin_memory 加速 CPU 到 GPU 的数据传输。
		x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
		# 数据集配置启用 Mixup/CutMix 时，同时混合图像和标签。
		if mixup_fn:
			x, y = mixup_fn(x, y)
		# 自动混合精度降低显存占用并加速 Tensor Core 计算。
		with torch.cuda.amp.autocast(enabled=config.misc.amp):
			if config.model.baseline_model:
				logits = model(x)
			else:
				logits = model(x, y)
		# MPSA 可能同时返回主损失和辅助损失，统一拆分处理。
		logits, loss, other_loss = loss_in_iters(logits, y, criterion)

		is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
		# 执行反向传播、可选梯度裁剪和优化器更新。
		grad_norm = loss_scaler(loss, optimizer, clip_grad=config.train.clip_grad,
		                        parameters=model.parameters(), create_graph=is_second_order)

		optimizer.zero_grad()
		# 本项目的余弦调度器按 iteration 更新，而非每个 epoch 更新。
		scheduler.step_update(global_step + 1)
		loss_scale_value = loss_scaler.state_dict()["scale"]

		# Mixup 标签不是单一类别，故只有未启用 Mixup 时统计训练准确率。
		if mixup_fn is None:
			preds = torch.argmax(logits, dim=-1)
			all_preds, all_label = save_preds(preds, y, all_preds, all_label)
		torch.cuda.synchronize()

		if grad_norm is not None:
			norm_meter.update(grad_norm)
		scaler_meter.update(loss_scale_value)
		loss_meter.update(loss.item(), y.size(0))

		lr = optimizer.param_groups[0]['lr']
		# 记录训练指标；辅助损失仅在模型确实返回时写入。
		if writer:
			writer.add_scalar("train/loss", loss_meter.val, global_step)
			writer.add_scalar("train/lr", lr, global_step)
			writer.add_scalar("train/grad_norm", norm_meter.val, global_step)
			writer.add_scalar("train/scaler_meter", scaler_meter.val, global_step)
			if other_loss:
				try:
					loss1_meter.update(other_loss[0].item(), y.size(0))
					loss2_meter.update(other_loss[1].item(), y.size(0))
					loss3_meter.update(other_loss[2].item(), y.size(0))
				except:
					pass

				writer.add_scalar("losses/t_loss", loss_meter.val, global_step)
				writer.add_scalar("losses/1_loss", loss1_meter.val, global_step)
				writer.add_scalar("losses/2_loss", loss2_meter.val, global_step)
				writer.add_scalar("losses/3_loss", loss3_meter.val, global_step)

		# set_postfix require dic input
		p_bar.set_postfix(loss="%2.5f" % loss_meter.avg, lr="%.5f" % lr, gn="%1.4f" % norm_meter.avg)
		p_bar.update()

	# 一个 epoch 结束后关闭进度条并汇总训练集准确率。
	p_bar.close()
	train_accuracy = eval_accuracy(all_preds, all_label, config) if mixup_fn is None else 0.0
	return train_accuracy


def loss_in_iters(output, targets, criterion):
	"""兼容普通分类器输出与 MPSA 的“logits + 多项损失”输出格式。"""
	if not isinstance(output, (list, tuple)):
		return output, criterion(output, targets), None
	else:
		logits, loss = output
		if not isinstance(loss, (list, tuple)):
			return logits, loss, None
		else:
			return logits, loss[0], loss[1:]

@torch.no_grad()
def valid(config, model, test_loader, epoch=-1, train_acc=0.0, writer=None,save_feature=False):
	"""在测试集上评估分类精度和交叉熵损失；可选保存特征供可视化使用。"""
	criterion = torch.nn.CrossEntropyLoss()
	# 启用评估模式并禁用梯度，降低验证阶段的显存和计算开销。
	model.eval()

	step_per_epoch = len(test_loader)
	p_bar = tqdm(total=step_per_epoch,
	             desc=f'Valid {(epoch + 1) // config.misc.eval_every:^3}/{math.ceil(config.train.epochs / config.misc.eval_every):^3}',
	             dynamic_ncols=True,
	             ascii=True,
	             disable=config.local_rank not in [-1, 0])

	loss_meter = AverageMeter()
	acc_meter = AverageMeter()
	saved_feature,saved_labels = [],[]
	for step, (x, y) in enumerate(test_loader):
		x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)

		with torch.cuda.amp.autocast(enabled=config.misc.amp):
			logits = model(x)

		# 断点恢复后的特征保存模式，用于后续 t-SNE 等可视化。
		if save_feature:
			saved_feature.append(logits)
			saved_labels.append(y)
		loss = criterion(logits, y.long())
		acc = accuracy(logits, y)[0]
		# 多卡验证时，对各进程的精度结果做归约。
		if config.local_rank != -1:
			acc = reduce_mean(acc)

		loss_meter.update(loss.item(), y.size(0))
		acc_meter.update(acc.item(), y.size(0))

		p_bar.set_postfix(acc="{:2.3f}".format(acc_meter.avg), loss="%2.5f" % loss_meter.avg,
		                  tra="{:2.3f}".format(train_acc * 100))
		p_bar.update()
		pass
	if save_feature:
		# 保存全部测试样本的输出特征和标签，不影响普通训练验证。
		os.makedirs('visualize/saved_features',exist_ok=True)
		saved_feature = torch.cat(saved_feature, 0)
		saved_labels = torch.cat(saved_labels,0)
		torch.save(saved_feature,f'visualize/saved_features/{config.data.dataset}_f.pth')
		torch.save(saved_labels, f'visualize/saved_features/{config.data.dataset}_l.pth')
	p_bar.close()
	if writer:
		writer.add_scalar("test/accuracy", acc_meter.avg, epoch + 1)
		writer.add_scalar("test/loss", loss_meter.avg, epoch + 1)
		writer.add_scalar("test/train_acc", train_acc * 100, epoch + 1)
	return acc_meter.avg, loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, log, rank):
	"""预热 GPU 后测量模型纯前向推理吞吐量（样本数/秒）。"""
	model.eval()
	for idx, (images, _) in enumerate(data_loader):
		images = images.cuda(non_blocking=True)
		batch_size = images.shape[0]
		for i in range(50):
			model(images)
		torch.cuda.synchronize()
		if rank in [-1, 0]:
			log.info(f"throughput averaged with 30 times")
		tic1 = time.time()
		for i in range(30):
			model(images)
		torch.cuda.synchronize()
		tic2 = time.time()
		if rank in [-1, 0]:
			log.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
		return


if __name__ == '__main__':
	# 从 setup.py 已准备好的全局配置启动训练。
	main(config)
