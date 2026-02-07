from sklearn.preprocessing import OneHotEncoder as OHE
import pandas as pd
import transformers
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from torch.optim import AdamW
from statistics import mean
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support as calculate_score
import logging
import os
from datetime import datetime
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
from esam_model import ESAMBertForSequenceClassification, ESAMDataProcessor

# 配置常量
CLASSES = [
   'T1003', 'T1005', 'T1012', 'T1016', 'T1021', 'T1027',
   'T1033', 'T1036', 'T1041', 'T1047', 'T1053', 'T1055',
   'T1056', 'T1057', 'T1059', 'T1068', 'T1070',
   'T1071', 'T1072', 'T1074', 'T1078', 'T1082', 'T1083',
   'T1090', 'T1095', 'T1105', 'T1106', 'T1110', 'T1112', 'T1113',
   'T1140', 'T1190', 'T1204', 'T1210', 'T1218', 'T1219',
   'T1484', 'T1518', 'T1543', 'T1547', 'T1548',
   'T1552', 'T1557', 'T1562', 'T1564', 'T1566',
   'T1569', 'T1570', 'T1573', 'T1574'
]

# 模型配置
MODEL_CONFIGS = {
    # 'BERT-Large': {
    #     'tokenizer_class': transformers.BertTokenizer,
    #     'model_class': transformers.BertForSequenceClassification,
    #     'model_name_or_path': 'bert-large-cased',
    #     'num_labels': len(CLASSES)
    # },
    # 'SecureBERT': {
    #     'tokenizer_class': transformers.RobertaTokenizer,
    #     'model_class': transformers.RobertaForSequenceClassification,
    #     'model_name_or_path': 'ehsanaghaei/SecureBERT',
    #     'num_labels': len(CLASSES)
    # },
    # 'SecBERT': {
    #     'tokenizer_class': transformers.AutoTokenizer,
    #     'model_class': transformers.AutoModelForSequenceClassification,
    #     'model_name_or_path': 'jackaduma/SecBERT',
    #     'num_labels': len(CLASSES)
    # },
    # 'SciBERT': {
    #     'tokenizer_class': transformers.BertTokenizer,
    #     'model_class': transformers.BertForSequenceClassification,
    #     'model_name_or_path': 'allenai/scibert_scivocab_uncased',
    #     'num_labels': len(CLASSES)
    # },
    'ESAM-SciBERT': {
        'tokenizer_class': transformers.BertTokenizer,
        'model_class': ESAMBertForSequenceClassification,
        'model_name_or_path': 'allenai/scibert_scivocab_uncased',
        'num_labels': len(CLASSES),
        'use_esam': True,
        'description_file': r'technique_description_f.csv'
    }
}

# 全局训练轮数常量（可全局修改）
EPOCHS = 4

class CtiModel:
    """CTI模型训练和评估的统一类"""

    def __init__(self, model_name: str, data_path: str = None):
        # 私有属性
        self._model_name = model_name
        self._cuda = None
        self._encoder = None
        self._tokenizer = None
        self._model = None
        self._data = None
        self._train_data = None
        self._test_data = None
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None
        self._logger = None

        # ESAM相关
        self._esam_processor = None
        self._description_emb_train = None
        self._description_emb_test = None

        # 输出目录结构
        self._output_dir = 'output'
        self._saved_models_dir = os.path.join(self._output_dir, 'saved_models')
        self._logs_dir = os.path.join(self._output_dir, 'logs')
        self._loss_dir = os.path.join(self._output_dir, 'loss')
        self._result_dir = os.path.join(self._output_dir, 'result')
        self._matrix_dir = os.path.join(self._output_dir, 'matrix')

        # 创建所有输出目录
        for dir_path in [self._saved_models_dir, self._logs_dir, self._loss_dir, self._result_dir, self._matrix_dir]:
            os.makedirs(dir_path, exist_ok=True)

        # 数据路径
        self._data_path = data_path or r'smallData/TRAM_fine_tuned_SciBERT copy.json'

        # 初始化
        self._setup_logging()
        self._setup_device()
        self._setup_data()

    def _setup_logging(self):
        """设置日志"""
        log_file = os.path.join(self._logs_dir, f'{self._model_name}_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self._logger = logging.getLogger(f'{self._model_name}')
        self._logger.info(f"初始化 {self._model_name} 模型")

    def _setup_device(self):
        """设置设备"""
        self._cuda = torch.device('cuda')
        self._logger.info(f"设备设置为: {self._cuda}")

    def _setup_data(self):
        """设置数据和编码器"""
        self._logger.info("开始编码阶段")
        self._encoder = OHE(sparse_output=False)
        self._encoder.fit([[c] for c in CLASSES])
        self._logger.info(f"类别数量: {len(CLASSES)}")

        # 加载数据
        self._data = pd.read_json(self._data_path)
        self._logger.info(f"数据加载完成: {len(self._data)} 条记录")

        # 分割数据
        self._train_data, self._test_data = train_test_split(self._data, test_size=0.2, shuffle=True)
        self._logger.info(f"数据分割完成: 训练集 {len(self._train_data)}, 测试集 {len(self._test_data)}")

    def load_model(self):
        """加载模型和分词器"""
        self._logger.info(f"加载 {self._model_name} 分词器和模型")

        config = MODEL_CONFIGS[self._model_name]
        tokenizer_class = config['tokenizer_class']
        model_class = config['model_class']
        model_name_or_path = config['model_name_or_path']
        num_labels = config['num_labels']

        try:
            if self._model_name == 'ESAM-SciBERT':
                # ESAM特殊处理
                self._tokenizer = tokenizer_class.from_pretrained(model_name_or_path)

                # 初始化ESAM数据处理器
                base_model = transformers.BertModel.from_pretrained(model_name_or_path)
                self._esam_processor = ESAMDataProcessor(
                    description_file=config['description_file'],
                    tokenizer=self._tokenizer,
                    model=base_model
                )

                # 创建ESAM模型
                self._model = model_class(
                    model_name_or_path=model_name_or_path,
                    num_classes=num_labels,
                    description_embeddings=self._esam_processor.description_embeddings
                )
            # elif self._model_name == 'SciBERT':
            #     # SciBERT特殊处理：从本地路径加载
            #     self._tokenizer = tokenizer_class.from_pretrained(model_name_or_path, max_length=512)
            #     model_path = r'scibert_single_label_model'
            #     self._model = model_class.from_pretrained(model_path, num_labels=num_labels)
            else:
                # 其他模型直接从HuggingFace加载
                self._tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
                self._model = model_class.from_pretrained(model_name_or_path, num_labels=num_labels)

            self._logger.info(f"{self._model_name} 模型加载完成")
        except Exception as e:
            self._logger.error(f"加载 {self._model_name} 模型失败: {e}")
            raise

        self._model = self._model.to(self._cuda).train()

    def prepare_data(self):
        """准备训练和测试数据"""
        # 编码标签
        self._y_train = self._encode_labels(self._train_data[['label']])
        self._y_test = self._encode_labels(self._test_data[['label']])

        # 分词文本
        self._x_train = self._tokenize(self._train_data['text'].tolist())
        self._x_test = self._tokenize(self._test_data['text'].tolist())

        # 为ESAM准备描述嵌入
        if self._model_name == 'ESAM-SciBERT' and self._esam_processor:
            self._description_emb_train = self._prepare_description_embeddings(self._train_data['label'].tolist())
            self._description_emb_test = self._prepare_description_embeddings(self._test_data['label'].tolist())

        self._logger.info(f"训练集编码完成: {self._x_train.shape}, {self._y_train.shape}")
        self._logger.info(f"测试集编码完成: {self._x_test.shape}, {self._y_test.shape}")

    def _prepare_description_embeddings(self, labels: list):
        """为ESAM准备描述嵌入"""
        embeddings = []
        for label in labels:
            emb = self._esam_processor.get_description_embedding(label)
            embeddings.append(emb)
        return torch.stack(embeddings).to(self._cuda)

    def _encode_labels(self, labels):
        """编码标签"""
        return torch.Tensor(self._encoder.transform(labels))

    def _tokenize(self, instances: list[str]):
        """分词文本"""
        return self._tokenizer(instances, return_tensors='pt', padding='max_length', truncation=True, max_length=512).input_ids

    def _load_data_batch(self, x, y, batch_size=16):
        """加载批次数据"""
        x_len, y_len = x.shape[0], y.shape[0]
        assert x_len == y_len
        for i in range(0, x_len, batch_size):
            slc = slice(i, i + batch_size)
            yield x[slc].to(self._cuda), y[slc].to(self._cuda)

    def train(self, num_epochs=EPOCHS, batch_size=16):
        """训练模型"""
        self._logger.info("----------------------训练阶段------------------------")
        self._logger.info(f"训练轮数: {num_epochs}, 批次大小: {batch_size}")

        loss_records = []
        epoch_loss_records = []
        optimizer = AdamW(self._model.parameters(), lr=2e-5, eps=1e-8)

        for epoch in range(num_epochs):
            epoch_losses = []
            self._logger.info(f"开始训练第 {epoch + 1} 轮")

            for batch_idx, (x, y) in enumerate(tqdm(self._load_data_batch(self._x_train, self._y_train, batch_size=batch_size))):
                self._model.zero_grad()

                # 准备输入
                inputs = {
                    'input_ids': x,
                    'attention_mask': x.ne(self._tokenizer.pad_token_id).to(int),
                    'labels': y
                }

                # 为ESAM添加描述嵌入
                if self._model_name == 'ESAM-SciBERT' and self._description_emb_train is not None:
                    # 获取当前批次的描述嵌入
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(self._description_emb_train))
                    desc_emb_batch = self._description_emb_train[start_idx:end_idx]
                    inputs['description_emb'] = desc_emb_batch
                    inputs['use_esam'] = True

                out = self._model(**inputs)
                loss = out['loss'].item()
                epoch_losses.append(loss)

                loss_records.append({
                    'epoch': epoch + 1,
                    'batch': batch_idx + 1,
                    'loss': loss
                })

                out['loss'].backward()
                optimizer.step()

            if epoch_losses:
                avg_loss = mean(epoch_losses)
            else:
                avg_loss = 0.0

            # 计算验证集（测试集）上的平均损失
            val_losses = []
            self._model.eval()
            with torch.no_grad():
                for i in range(0, self._x_test.shape[0], batch_size):
                    x_val = self._x_test[i : i + batch_size].to(self._cuda)
                    y_val = self._y_test[i : i + batch_size].to(self._cuda)

                    val_inputs = {
                        'input_ids': x_val,
                        'attention_mask': x_val.ne(self._tokenizer.pad_token_id).to(int),
                        'labels': y_val
                    }

                    if self._model_name == 'ESAM-SciBERT' and self._description_emb_test is not None:
                        desc_emb_batch = self._description_emb_test[i : i + batch_size]
                        val_inputs['description_emb'] = desc_emb_batch
                        val_inputs['use_esam'] = True

                    val_out = self._model(**val_inputs)
                    # 这里假定模型返回字典包含 'loss'
                    val_losses.append(val_out['loss'].item())

            if val_losses:
                val_avg = mean(val_losses)
            else:
                val_avg = 0.0

            epoch_loss_records.append({'epoch': epoch + 1, 'avg_loss': avg_loss, 'val_loss': val_avg})
            self._logger.info(f"Epoch {epoch + 1} train_avg: {avg_loss:.4f} val_avg: {val_avg:.4f}")
            print(f"Epoch {epoch + 1} train_avg: {avg_loss:.4f} val_avg: {val_avg:.4f}")

        # # 保存训练损失
        # loss_df = pd.DataFrame(loss_records)
        # loss_csv_path = os.path.join(self._loss_dir, f'{self._model_name}_loss_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        # loss_df.to_csv(loss_csv_path, index=False)
        # self._logger.info(f"训练损失已保存到: {loss_csv_path}")

        # 保存每个 epoch 的平均 loss
        epoch_loss_df = pd.DataFrame(epoch_loss_records)
        epoch_loss_csv_path = os.path.join(self._loss_dir, f'{self._model_name}_epoch_loss_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        epoch_loss_df.to_csv(epoch_loss_csv_path, index=False)
        self._logger.info(f"每轮平均损失已保存到: {epoch_loss_csv_path}")

    def evaluate(self, batch_size=16):
        """评估模型"""
        self._logger.info("----------------------评估阶段------------------------")
        self._model.eval()

        preds = []
        with torch.no_grad():
            for i in range(0, self._x_test.shape[0], batch_size):
                x = self._x_test[i : i + batch_size].to(self._cuda)

                # 准备输入
                inputs = {
                    'input_ids': x,
                    'attention_mask': x.ne(self._tokenizer.pad_token_id).to(int)
                }

                # 为ESAM添加描述嵌入
                if self._model_name == 'ESAM-SciBERT' and self._description_emb_test is not None:
                    desc_emb_batch = self._description_emb_test[i : i + batch_size]
                    inputs['description_emb'] = desc_emb_batch
                    inputs['use_esam'] = True

                out = self._model(**inputs)
                preds.extend(out['logits'].to('cpu'))

        # 处理预测结果
        predicted_labels = (
            self._encoder.inverse_transform(
                F.one_hot(
                    torch.vstack(preds).softmax(-1).argmax(-1),
                    num_classes=len(self._encoder.categories_[0])
                ).numpy()
            ).reshape(-1)
        )

        predicted = list(predicted_labels)
        actual = self._test_data['label'].tolist()

        # 收集错误样本信息（真实标签 != 预测标签）
        errors = []
        texts = self._test_data['text'].tolist()
        for i, (act, pred) in enumerate(zip(actual, predicted)):
            if act != pred:
                errors.append({'index': i, 'text': texts[i], 'actual': act, 'predicted': pred})

        # 统计每个标签的 FP（被预测为该标签但真实不是）和 FN（真实为该标签但预测不是）
        fp_counts = {label: 0 for label in CLASSES}
        fn_counts = {label: 0 for label in CLASSES}
        for e in errors:
            # predicted is false positive for that predicted label
            if e['predicted'] in fp_counts:
                fp_counts[e['predicted']] += 1
            # actual is false negative for that actual label
            if e['actual'] in fn_counts:
                fn_counts[e['actual']] += 1

        # 取前五个最多的 FP 和 FN
        top5_fp = sorted(fp_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top5_fn = sorted(fn_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # 保存详情到文件
        issues_path = os.path.join(self._result_dir, f'{self._model_name}_top5_FP_FN_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        with open(issues_path, 'w', encoding='utf-8') as fout:
            fout.write(f"timestamp: {datetime.now().isoformat()}\n")
            fout.write(f"model: {self._model_name}\n\n")

            fout.write("Top5 False Positives (predicted label -> count):\n")
            for label, cnt in top5_fp:
                fout.write(f"{label}: {cnt}\n")
                # 列出所有对应的样本
                for e in errors:
                    if e['predicted'] == label:
                        fout.write(f"  - idx: {e['index']}, true: {e['actual']}, pred: {e['predicted']}, text: {e['text']}\n")
                fout.write("\n")

            fout.write("\nTop5 False Negatives (actual label -> count):\n")
            for label, cnt in top5_fn:
                fout.write(f"{label}: {cnt}\n")
                for e in errors:
                    if e['actual'] == label:
                        fout.write(f"  - idx: {e['index']}, true: {e['actual']}, pred: {e['predicted']}, text: {e['text']}\n")
                fout.write("\n")

        self._logger.info(f"前五个 FP/FN 详情已保存到: {issues_path}")
        print(f"前五个 FP/FN 详情已保存到: {issues_path}")

        labels = sorted(set(actual) | set(predicted))
        scores = calculate_score(actual, predicted, labels=labels)

        scores_df = pd.DataFrame(scores).T
        scores_df.columns = ['P', 'R', 'F1', '#']
        scores_df.index = labels
        scores_df.loc['(micro)'] = calculate_score(actual, predicted, average='micro', labels=labels)
        scores_df.loc['(macro)'] = calculate_score(actual, predicted, average='macro', labels=labels)

        # 计算混淆矩阵指标
        mcm = multilabel_confusion_matrix(actual, predicted, labels=labels)
        tp_list = []
        fp_list = []
        tn_list = []
        fn_list = []
        for cm in mcm:
            tn, fp, fn, tp = cm.ravel()
            tp_list.append(tp)
            fp_list.append(fp)
            tn_list.append(tn)
            fn_list.append(fn)

        # 添加到 DataFrame
        scores_df['TP'] = tp_list + [None, None]
        scores_df['FP'] = fp_list + [None, None]
        scores_df['TN'] = tn_list + [None, None]
        scores_df['FN'] = fn_list + [None, None]

        # 更新列名
        scores_df.columns = ['P', 'R', 'F1', '#', 'TP', 'FP', 'TN', 'FN']

        # 保存评估结果
        eval_csv_path = os.path.join(self._result_dir, f'{self._model_name}_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        with open(eval_csv_path, 'w', encoding='utf-8') as f:
            # 写入元数据
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'model': self._model_name,
                'num_classes': len(CLASSES),
                'train_size': len(self._train_data),
                'test_size': len(self._test_data),
                'epochs': EPOCHS,
                'batch_size': 16
            }
            for key, value in metadata.items():
                f.write(f"# {key}: {value}\n")
            f.write("\n")
            scores_df.to_csv(f)

        self._logger.info(f"评估结果已保存到: {eval_csv_path}")
        print("评估结果：")
        print(scores_df)

        # 计算并保存全局混淆矩阵
        cm = confusion_matrix(actual, predicted, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)

        cm_csv_path = os.path.join(self._matrix_dir, f'{self._model_name}_confusion_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        cm_df.to_csv(cm_csv_path)

        self._logger.info(f"混淆矩阵已保存到: {cm_csv_path}")
        # print("\n混淆矩阵：")
        # print(cm_df)

    def save_model(self):
        """保存模型"""
        self._logger.info("----------------------保存模型阶段------------------------")

        model_save_path = os.path.join(self._saved_models_dir, f'{self._model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_model')
        tokenizer_save_path = os.path.join(self._saved_models_dir, f'{self._model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}_tokenizer')

        if self._model_name == 'ESAM-SciBERT':
            # ESAM模型特殊保存
            torch.save({
                'model_state_dict': self._model.state_dict(),
                'bert_config': self._model.bert.config,
                'esam_config': {
                    'hidden_size': self._model.hidden_size,
                    'num_classes': self._model.num_classes
                },
                'description_embeddings': self._model.description_embeddings
            }, f"{model_save_path}.pth")
        else:
            self._model.save_pretrained(model_save_path)

        self._tokenizer.save_pretrained(tokenizer_save_path)

        self._logger.info(f"模型已保存到: {model_save_path}")
        self._logger.info(f"分词器已保存到: {tokenizer_save_path}")

    def run_experiment(self, num_epochs=EPOCHS, train_batch_size=16, test_batch_size=16):
        """运行完整实验"""
        try:
            self.load_model()
            self.prepare_data()
            self.train(num_epochs=num_epochs, batch_size=train_batch_size)
            self.evaluate(batch_size=test_batch_size)
            # self.save_model()
            self._logger.info(f"{self._model_name} 实验完成")
            return True
        except Exception as e:
            self._logger.error(f"{self._model_name} 实验失败: {e}")
            return False

def main():
    """主函数：运行所有模型的对比实验"""
    print("开始模型对比实验")

    # 运行所有模型的实验
    for model_name in MODEL_CONFIGS.keys():
        print(f"\n{'='*60}")
        print(f"开始 {model_name} 实验")
        print(f"{'='*60}")

        try:
            # 创建模型实例
            model = CtiModel(model_name)

            # 运行实验
            success = model.run_experiment(num_epochs=EPOCHS, train_batch_size=16, test_batch_size=16)

            if success:
                print(f"{model_name} 实验成功完成")
            else:
                print(f"{model_name} 实验失败")

        except Exception as e:
            print(f"{model_name} 实验出现异常: {e}")
            continue

    print(f"\n{'='*60}")
    print("所有模型对比实验完成")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
    
