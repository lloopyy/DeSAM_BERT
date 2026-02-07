import copy
import sys
import traceback

import main as main_mod
from esam_model import ESAMBertForSequenceClassification

"""
esam_compare.py

与 compare.py 相同的行为：把 BERT-Large、SecureBERT、SecBERT、SciBERT 转成 ESAM 版本并运行主流程。
保留 ESAM-SciBERT 不变，保持其它输出行为一致。
"""


def build_modified_configs():
    modified = copy.deepcopy(main_mod.MODEL_CONFIGS)
    targets = ['BERT-Large', 'SecureBERT', 'SecBERT', 'SciBERT']
    for name in targets:
        if name in modified:
            modified[name]['model_class'] = ESAMBertForSequenceClassification
            modified[name]['use_esam'] = True
            if 'description_file' not in modified[name]:
                modified[name]['description_file'] = r'technique_description_f.csv'
    return modified


def run():
    try:
        print('准备替换模型为 ESAM 版本并运行实验...')
        modified = build_modified_configs()
        main_mod.MODEL_CONFIGS = modified
        main_mod.main()
    except Exception:
        print('执行失败:')
        traceback.print_exc()


if __name__ == '__main__':
    run()
