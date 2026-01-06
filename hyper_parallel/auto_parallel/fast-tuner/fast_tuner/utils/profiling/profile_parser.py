# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""profile parser"""
import json
import os
import statistics
import csv
from collections import defaultdict
from pathlib import Path
from fast_tuner.pipeline_conductor.pp_util import parse_shell
from fast_tuner.utils.logger import logger
from fast_tuner.ndsearch.para_for_nd_search import ParaForNd
from fast_tuner.ndsearch.memory_model import compute_weight_and_optimizer_memory

encoding = 'utf-8'

def find_file_by_name(directory, filename):
    for root, _, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    logger.error(f'No such file {filename} in directory {directory}')
    return None

def median_mean(durs):
    if len(durs) == 0:
        logger.info("get durs no values")
        return 0
    median_durs = durs[len(durs)//4 : len(durs) - len(durs)//4]
    return round(statistics.mean(median_durs)/1000, 3)

class ProfileParser:
    '''
    The recommend config for different pp is：
    pp2: [6,5] recomp [6,0], reduce laysers when OOM, layers at least [5,4]
    if still OOM, do another profile wrt [3,1] no recomp w/ dense_replace = 1
    pp4: [3,3,3,2] recomp [3,3,0,0]
    pp8: [3,1,1,1,1,2,2,2] recomp [3,1,1,1,1,2,0,2] (or try pp4 with layers [3,2,2,2] w/ recomp [3,2,0,2])
    a typical input is a string and a tuple of numbers. 2-tuple for pp2 and 4-tuple for higher pp,
    which consists the rank for each stages.
    '''

    profile_data = None
    config = ''
    # recomp_bws, moe_fws1, moe_fws2, moe_bws, dense_fws, dense_bws, totals = [], [], [], [], [], [], []
    # moe_fw1, moe_fw2, moe_bw, recomp_bw, dense_fw, dense_bw, head_ratio, num_last_stage_layers, mtp = 0
    dense_fws, dense_bws, recomp_dense_bws, moe_fws, moe_bws, recomp_moe_bws, totals = [], [], [], [], [], [], []
    (dense_fw, dense_bw, recomp_dense_bw, moe_fw, moe_bw,
     recomp_moe_bw, total, head_ratio, num_last_stage_layers, mtp) = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    att_tid, grad_att_tid = 2, 2

    def __init__(self, input_args, para):
        self.para = para
        self.mbn = input_args.mbn
        self.layer_num = input_args.num_layers
        self.num_layer_list = input_args.num_layer_list
        self.recompute_num_layers = input_args.recompute_num_layers
        self.dense_num = input_args.first_k_dense_replace
        self.step = input_args.profile_steps
        self.model_name = input_args.model_name

    def parse_batch_profile_result(self, profile_configs_results):
        """解析批profile文件"""
        # step1. 解析profile文件
        # ep填1, dmratrion:0.33, bfratio=dense_bw/dense_fw, re_grow_ration = 1,hratio=0.68
        # [dp, tp, pp, dmratio, bfratio, re_grow_ration, hratio, moe_fw]
        profile_result = []
        dense_flag = False
        #Todo 这里para.YAML_PATH的实现待修改
        if self.para.YAML_PATH:
            for result in profile_configs_results:
                profile_dir = result[-1]
                pp = result[2]
                self.layer_num = result[-2]
                if len(result) > 6:
                    config = result[:4]
                else:
                    config = result[:3]
                config += [0.33, 0, 1, 0.68, 0]
                self.config_anal_func(profile_dir, pp, config)
                self.refresh()
                profile_result.append(config)
        elif self.para.SHELL_PATH:
            for result in profile_configs_results:
                profile_dir = result[-1]
                pp = result[2]
                update_input_args, _ = parse_shell(result[-2])
                self.mbn = update_input_args.get('GBS') // update_input_args.get('MBS') // update_input_args.get('DP')
                self.layer_num = update_input_args.get('NUM_LAYERS')
                if 'FIRST_K_DENSE_RAPLACE' in update_input_args:
                    self.dense_num = update_input_args.get('FIRST_K_DENSE_RAPLACE')
                else:
                    self.dense_num = 3
                logger.info(f"mbn: {self.mbn}, layer_num: {self.layer_num}.")
                if len(result) > 8:
                    config = result[:4]
                else:
                    config = result[:3]
                config += [0.33, 0, 1, 0.68, 0]
                self.config_anal_func(profile_dir, pp, config)
                self.refresh()
                profile_result.append(config)
        elif self.para.TOML_PATH:
            for result in profile_configs_results:
                profile_dir = result[-1]
                pp = result[2]
                config = result[:-2]
                config += [0.33, 1, 1, 0.68, 16]
                self.config_anal_func(profile_dir, pp, config)
                self.refresh()
                profile_result.append(config)

        # step2. 生成csv文件
        # dense模型的result长度是8，moe模型的result长度是9
        if len(profile_result[0]) == 8:
            dense_flag = True
        self.write_result_to_csv(profile_result, dense_flag)

    def write_result_to_csv(self, profile_result, is_llama=False):
        """"把profile结果写入csv"""
        if self.para.YAML_PATH or self.para.SHELL_PATH:
            if is_llama:
                headers = ['dp', 'tp', 'pp', 'dmratio', 'bfratio', 're_grow_ration', 'hratio', 'moe_fw']
            else:
                headers = ['dp', 'tp', 'pp', 'ep', 'dmratio', 'bfratio', 're_grow_ration', 'hratio', 'moe_fw']
        else:
            headers = ['dp', 'tp', 'pp', 'ep', 'cp', 'op', 'dmratio', 'bfratio', 're_grow_ration', 'hratio', 'moe_fw']
        # 写入 CSV 文件
        try:
            csv_path = os.path.join(os.path.abspath(self.para.OUTPUT_PATH), 'profile_parser_result.csv')
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                # 写入表头
                writer.writerow(headers)
                # 写入数据
                writer.writerows(profile_result)
            logger.info(f"CSV file {csv_path} generate succ.")
            self.para.args.parser_result = csv_path
        except Exception as e:
            print(f"write CSV file fail: {e}")

    def load_profile(self, file, rk):
        """读取profile文件trace view"""        
        path = os.path.abspath('.\\profile_info') + '\\' + str(file)
        path = find_file_by_name(path + '\\rank_' + str(rk), 'trace_view.json')
        with open(path, 'r', encoding=encoding) as f:
            self.profile_data = json.load(f)

    def load_profile_by_dir(self, rank_dir):
        """从文件夹读取profile文件kernel details"""
        path = find_file_by_name(rank_dir, 'trace_view.json')
        with open(path, 'r', encoding=encoding) as f:
            self.profile_data = json.load(f)

    def load_profile_by_kernel(self, file):
        """读取profile文件kernel details"""
        path = find_file_by_name(file, 'kernel_details.csv')
        if path is None:
            logger.error(f"can not find kernel details file {file}")
            return False
        logger.info(f'path of kernel_details is {path}')
        with open(path, 'r', encoding=encoding) as f:
            reader = csv.DictReader(f)
            self.profile_data = list(reader)
        return True

    def load_profile_no_recompute(self, file, rk):
        """读取profile文件不重算算子"""
        path = os.path.abspath('.\\profile_info') + '\\' + str(file)
        path = find_file_by_name(path + '\\rank_' + str(rk) + '_norecomp' , 'trace_view.json')
        with open(path, 'r', encoding=encoding) as f:
            self.profile_data = json.load(f)

    def refresh(self):
        self.profile_data = []
        (self.dense_fws, self.dense_bws, self.recomp_dense_bws, self.moe_fws, self.moe_bws,
         self.recomp_moe_bws, self.totals) = [
            [] for _ in range(7)]
        (self.dense_fw, self.dense_bw, self.recomp_dense_bw, self.moe_fw, self.moe_bw, self.recomp_moe_bw,
         self.total, self.head_ratio, self.num_last_stage_layers, self.mtp) = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    def release(self):
        self.profile_data = []

    def set_tid(self, a, g):
        self.att_tid, self.grad_att_tid = a, g

    def extract_atts(self):
        atts, grad_atts = [], []
        for line in self.profile_data:
            name = line['name']
            if line['tid'] == self.att_tid and name.endswith('FlashAttentionScore_FlashAttentionScore'):
                atts.append(float(line['ts']))
            if line['tid'] == self.grad_att_tid and name.endswith('FlashAttentionScoreGrad_FlashAttentionScoreGrad'):
                grad_atts.append(float(line['ts']))
        logger.info(f'num of atts is {len(atts)}, num of grad_atts is {len(grad_atts)}')
        return atts, grad_atts

    def extract_atts_by_kernel(self):
        """从kernel details文件读取attributes"""
        atts, grad_atts = [], []
        try:
            with open(self.profile_data, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    name_value = row.get('Name', '')
                    if isinstance(name_value, str):
                        if name_value.endswith('FlashAttentionScore_FlashAttentionScore'):
                            ts_value = float(row.get('Start Time(us)'))
                            atts.append(ts_value)
                        elif name_value.endswith('FlashAttentionScoreGrad_FlashAttentionScoreGrad'):
                            ts_value = float(row.get('Start Time(us)'))
                            grad_atts.append(ts_value)
            logger.info(f'extract by kernel_details.csv num of atts is {len(atts)}, num of grad_atts')
            return atts, grad_atts
        except Exception as e:
            logger.error(f'extract by kernel_details.csv fail: {e}')
            return [], []

    def extract_atts_by_loaded_data(self):
        """读取profile文件kernel details"""
        atts, grad_atts = [], []
        try:
            for row in self.profile_data:
                if row['Type'] == 'FlashAttentionScore':
                    atts.append(float(row['Start Time(us)']))
                if row['Type'] == 'FlashAttentionScoreGrad':
                    grad_atts.append(float(row['Start Time(us)']))
            logger.info(f'num of atts is {len(atts)}, num of grad_atts is {len(grad_atts)}')
            return atts, grad_atts
        except Exception as e:
            logger.error(f'extract by loaded_data fail: {e}')
            return [], []

    def extract_atts_for_titan(self):
        """
        Docstring for extract_atts_for_titan
        
        :param self: Description
        """
        atts, grad_atts = [], []
        try:
            for row in self.profile_data:
                if row['Name'] == 'aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore':
                    atts.append(float(row['Start Time(us)']))
                if row['Name'] == 'aclnnFlashAttentionScoreGrad_FlashAttentionScoreGrad_FlashAttentionScoreGrad':
                    grad_atts.append(float(row['Start Time(us)']))
            logger.info(f'num of atts is {len(atts)}, num of grad_atts is {len(grad_atts)}')
            return atts, grad_atts
        except Exception as e:
            logger.error(f'extract by loaded_data fail: {e}')
            return [], []

    #todo: 待修改
    def stage_anal(self, pp, stage): #assuming we profiled 2 steps w/ micro = 32
        """
        Docstring for stage_anal
        
        :param self: Description
        :param pp: Description
        :param stage: Description
        """
        atts, grad_atts = self.extract_atts_by_kernel()
        layers, xlayers = len(grad_atts) // 64,  2 * len(grad_atts) // 64
        warm_up = (pp-1-stage) * layers
        atts = atts[warm_up : 32*xlayers - warm_up] + atts[32*xlayers + warm_up : 32*xlayers + 32*xlayers-warm_up]
        grad_atts = (grad_atts[warm_up : 32*layers - warm_up]
                     + grad_atts[32*layers + warm_up : 32*layers + 32*layers-warm_up])
        att_chunks = [atts[xlayers*i:xlayers*(i+1)] for i in range(len(atts)//xlayers-1)]
        grad_chunks = [grad_atts[layers*i:layers*(i+1)] for i in range(len(grad_atts)//layers-1)]
        if pp == 2:
            self.pp2_analysis(att_chunks, grad_chunks, layers, pp, stage)
        else:
            self.pp_not2_analysis(att_chunks, grad_chunks, layers, pp, stage)

    def pp_not2_analysis(self, att_chunks, grad_chunks, layers, pp, stage):
        """
        Docstring for pp_not2_analysis
        
        :param self: Description
        :param att_chunks: Description
        :param grad_chunks: Description
        :param layers: Description
        :param pp: Description
        :param stage: Description
        """
        if stage == 0:
            for chunk in att_chunks:
                for i in range(layers - 1):
                    if i <= 2:
                        self.dense_fws.append(chunk[i + 1] - chunk[i])
            for chunk in grad_chunks:
                for i in range(layers - 3, layers - 1):
                    self.dense_bws.append(chunk[i + 1] - chunk[i])
        elif stage == pp - 3:
            for chunk in att_chunks:
                for i in range(layers - 1):
                    self.moe_fws1.append(chunk[i + 1] - chunk[i])
            for chunk in grad_chunks:
                for i in range(layers - 3, layers - 1):
                    self.moe_bws.append(chunk[i + 1] - chunk[i])
        elif stage == pp - 2:
            for chunk in att_chunks:
                for i in range(layers - 1):
                    self.moe_fws2.append(chunk[i + 1] - chunk[i])
            for chunk in grad_chunks:
                for i in range(layers - 3, layers - 1):
                    self.moe_bws.append(chunk[i + 1] - chunk[i])
        else:
            for i in range(len(att_chunks) - 1):
                self.totals.append(att_chunks[i + 1][0] - att_chunks[i][0])
            self.num_last_stage_layers = layers

    def pp2_analysis(self, att_chunks, grad_chunks, layers, pp, stage):
        """
        Docstring for pp2_analysis
        
        :param self: Description
        :param att_chunks: Description
        :param grad_chunks: Description
        :param layers: Description
        :param pp: Description
        :param stage: Description
        """
        if stage == 0:
            for chunk in att_chunks:
                for i in range(layers - 1):
                    if i <= 2:
                        self.dense_fws.append(chunk[i + 1] - chunk[i])
                    else:
                        self.moe_fws1.append(chunk[i + 1] - chunk[i])
            for chunk in grad_chunks:
                for i in range(layers - 3, layers - 1):
                    self.dense_bws.append(chunk[i + 1] - chunk[i])
                for i in range(layers - 4):
                    self.recomp_bws.append(chunk[i + 1] - chunk[i])
            for i in range(len(att_chunks) - 1):
                self.totals.append(att_chunks[i + 1][0] - att_chunks[i][0])
        if stage == pp - 1:
            for chunk in att_chunks:
                for i in range(layers - 2):
                    self.moe_fws2.append(chunk[i + 1] - chunk[i])
            for chunk in grad_chunks:
                for i in range(2, layers - 1):
                    self.moe_bws.append(chunk[i + 1] - chunk[i])
            for i in range(len(att_chunks) - 1):
                self.totals.append(att_chunks[i + 1][0] - att_chunks[i][0])
            self.num_last_stage_layers = layers

    def stage_anal_for_pp1(self):
        """
        Docstring for stage_anal_for_pp1
        
        :param self: Description
        """
        atts, grad_atts = self.extract_atts_for_titan()
        layer_num = 10  # 与titan的__init__.py中的
        try:
            # pylint: disable=C0415
            import torchtitan.protocols.train_spec as train_spec_module
            model_flavor = 'tune'
            train_spec = train_spec_module.get_train_spec(self.model_name)
            model_args = train_spec.model_args[model_flavor]
            layer_num = model_args.n_layers
        except Exception as e:
            print(f'Error is: {e}, and do not get layer_num for profile parser.')

        atts_per_step = len(atts) // self.step
        grad_atts_per_step = len(grad_atts) // self.step
        att_chunks = [atts[atts_per_step * i:atts_per_step * (i + 1)] for i in range(self.step)]
        grad_chunks = [grad_atts[grad_atts_per_step * i:grad_atts_per_step * (i + 1)] for i in range(self.step)]

        for chunk in att_chunks:
            for i in range(layer_num - 1):
                self.moe_fws.append(chunk[i + 1] - chunk[i])

        for chunk in grad_chunks:
            for i in range(layer_num - 1):
                self.moe_bws.append(chunk[i + 1] - chunk[i])

    def stage_anal_for_pp2(self, pp, stage):
        """
        Docstring for stage_anal_for_pp2
        
        :param self: Description
        :param pp: Description
        :param stage: Description
        """
        atts, grad_atts = self.extract_atts_by_loaded_data()
        layer_num = self.num_layer_list[stage]
        warm_up = (pp - 1 - stage) * layer_num
        atts_per_step = len(atts) // self.step
        grad_atts_per_step = len(grad_atts) // self.step

        att_chunks = [atts[atts_per_step * i:atts_per_step * (i + 1)] for i in range(self.step)]
        grad_chunks = [grad_atts[grad_atts_per_step * i:grad_atts_per_step * (i + 1)] for i in range(self.step)]

        if pp == 2:
            if stage == 0:
                for chunk in att_chunks:
                    for i in range(self.mbn - 1):
                        idx = warm_up + i * (layer_num + self.recompute_num_layers)
                        for j in range(self.dense_num):
                            self.dense_fws.append(chunk[idx + j + 1] - chunk[idx + j])

                for chunk in grad_chunks:
                    for i in range(self.mbn - 2):
                        idx = warm_up + i * layer_num
                        self.dense_bws.append(chunk[idx + 2] - chunk[idx + 1])
                        self.recomp_dense_bws.append(chunk[idx + 3] - chunk[idx + 2])

            if stage == 1:
                for chunk in att_chunks:
                    for i in range(self.mbn):
                        idx = warm_up + i * (layer_num + self.recompute_num_layers)
                        for j in range(layer_num - 1):
                            self.moe_fws.append(chunk[idx + j + 1] - chunk[idx + j])
                        if i > 1:
                            idx_last = warm_up + (i - 1) * (layer_num + self.recompute_num_layers)
                            self.totals.append(chunk[idx] - chunk[idx_last])

                for chunk in grad_chunks:
                    for i in range(self.mbn):
                        idx = warm_up + i * layer_num
                        self.moe_bws.append(chunk[idx + 1] - chunk[idx])
                        self.recomp_moe_bws.append(chunk[idx + 2] - chunk[idx + 1])
                self.num_last_stage_layers = layer_num

    def config_anal(self, config, ranks):
        """
        Docstring for config_anal
        
        :param self: Description
        :param config: Description
        :param ranks: Description
        """
        self.config = config
        pp = len(ranks)
        for i in range(pp):
            self.load_profile(config, ranks[i])
            self.stage_anal(pp, i)
            self.release()

    def config_anal_func(self, profile_dir, pp, profile_result):
        """
        Docstring for config_anal_func
        
        :param self: Description
        :param profile_dir: Description
        :param pp: Description
        :param profile_result: Description
        """
        self.config = profile_dir
        self.process_folders(profile_dir, pp, profile_result)

    def process_folders(self, profile_result_dir, pp, profile_result):
        """
        遍历根目录下的所有文件夹，并使用 parser 函数解析每个文件夹中的内容

        参数:
        profile_result_dir (str): 根目录路径
        """
        root_path = Path(profile_result_dir)
        if not root_path.is_dir():
            logger.info(f"profile_result_dir:{profile_result_dir} not exist")
            return

        # 获取所有子文件夹--每个rank的文件夹
        folders = [f for f in root_path.iterdir() if f.is_dir()]
        folders.sort()
        i = 0
        for rank_folder in folders:
            logger.info(f"parsing: {rank_folder}")
            # Todo para.YAML_PATH的分支待修改
            if self.para.YAML_PATH:
                self.load_profile_by_dir(rank_folder)
                self.stage_anal(pp, i)
            else:
                find_file = self.load_profile_by_kernel(rank_folder)
                if find_file is False:
                    continue
                if self.para.SHELL_PATH:
                    self.stage_anal_for_pp2(pp, i)
                elif self.para.TOML_PATH:
                    self.stage_anal_for_pp1()
            self.release()
            i += 1
        self.refined_data()
        self.fill_result(profile_result)

    def data_display(self):
        """
        Docstring for data_display
        
        :param self: Description
        """
        logger.info(f'moe_fws1 are {self.moe_fws1}')
        logger.info(f'moe_fws2 are {self.moe_fws2}')
        logger.info(f'moe_bws is {self.moe_bws}')
        logger.info(f'recomp_bws is {self.recomp_bws}')
        logger.info(f'dense_fw is {self.dense_fw}')
        logger.info(f'dense_bw is {self.dense_bws}')
        logger.info(f'totals is {self.totals}')

    def refined_data(self):
        """
        Docstring for refined_data
        
        :param self: Description
        """
        if self.para.YAML_PATH or self.para.SHELL_PATH:
            if len(self.dense_fws) == 0 or len(self.dense_bws) == 0:
                logger.info(f"dense_fws len {len(self.dense_fws)} "
                            f"or dense_bws {len(self.dense_fws)} is empty")
                return
            self.dense_fw = median_mean(self.dense_fws)
            self.dense_bw = median_mean(self.dense_bws)
            self.recomp_dense_bw = median_mean(self.recomp_dense_bws)
            self.moe_fw = median_mean(self.moe_fws)
            self.moe_bw = median_mean(self.moe_bws)
            self.recomp_moe_bw = median_mean(self.recomp_moe_bws)
            self.total = median_mean(self.totals)
            self.mtp = (self.total - self.num_last_stage_layers * self.moe_fw -
                        (self.num_last_stage_layers - self.recompute_num_layers) * self.moe_bw - self.recomp_moe_bw)
            logger.info(
                f'config is {self.config}, dense forward is {self.dense_fw}, '
                f'dense backward is {self.dense_bw}, recompute dense backward is {self.recomp_dense_bw}, '
                f'moe forward is {self.moe_fw}, moe backward is {self.moe_bw}, '
                f'recompute moe backward is {self.recomp_moe_bw}, lmhead and MTP is {self.mtp}')
        elif self.para.TOML_PATH:
            if len(self.moe_fws) == 0 or len(self.moe_bws) == 0:
                logger.info(f"fws len {len(self.moe_fws)} "
                            f"or bws {len(self.moe_bws)} is empty")
                return
            self.moe_fw = median_mean(self.moe_fws)
            self.moe_bw = median_mean(self.moe_bws)
            logger.info(
                f'config is {self.config}, forward is {self.moe_fw}, backward is {self.moe_bw}')

    def fill_result(self, profile_result):
        """
        Docstring for fill_result
        
        :param self: Description
        :param profile_result: Description
        """
        if self.para.YAML_PATH or self.para.SHELL_PATH:
            if len(self.dense_fws) == 0 or len(self.dense_bws) == 0:
                logger.info(f"config {self.config} parse profile result is empty ")
                return
            profile_result[-1] = self.moe_fw
            profile_result[-2] = self.mtp / self.moe_fw
            profile_result[-3] = (self.recomp_moe_bw - self.moe_bw) / self.moe_fw
            profile_result[-4] = self.moe_bw / self.moe_fw
            profile_result[-5] = self.dense_fw / self.moe_fw
        elif self.para.TOML_PATH:
            if len(self.moe_fws) == 0 or len(self.moe_bws) == 0:
                logger.info(f"config {self.config} parse profile result is empty ")
                return
            profile_result[-1] = self.moe_fw
            profile_result[-2] = 1.5
            profile_result[-3] = 1.0
            profile_result[-4] = self.moe_bw / self.moe_fw
            profile_result[-5] = 0.3

class MemoryInfo:
    """
    Memory information
    """
    select_mem0 = 76
    select_mem12 = 76
    select_mem = 77
    re_comp_mem0 = 3
    re_comp_mem12 = 3
    re_comp_mem = 5
    act_mem0 = 79
    act_mem12 = 79
    act_mem = 79
    layer_mem012 = 349
    layer_mem = 340
    static_mem0 = 734
    static_mem = 116
    lm_head_mem = 690

    def write_to_txt(self, memory_file_name):
        """
        Docstring for write_to_txt
        
        :param self: Description
        :param memory_file_name: Description
        """
        with open(memory_file_name, 'w', encoding='utf-8') as file:
            file.write(f'select_mem0={self.select_mem0}\n')
            file.write(f'select_mem12={self.select_mem12}\n')
            file.write(f'select_mem={self.select_mem}\n')
            file.write(f're_comp_mem0={self.re_comp_mem0}\n')
            file.write(f're_comp_mem12={self.re_comp_mem12}\n')
            file.write(f're_comp_mem={self.re_comp_mem}\n')
            file.write(f'act_mem0={self.act_mem0}\n')
            file.write(f'act_mem12={self.act_mem12}\n')
            file.write(f'act_mem={self.act_mem}\n')
            file.write(f'layer_mem012={self.layer_mem012}\n')
            file.write(f'layer_mem={self.layer_mem}\n')
            file.write(f'static_mem0={self.static_mem0}\n')
            file.write(f'static_mem={self.static_mem}\n')
            file.write(f'lm_head_mem={self.lm_head_mem}\n')
            logger.info(f'Write memory info to {memory_file_name}')

class ProfileMemParser:
    """
    Docstring for ProfileMemParser
    """
    pipeline_output_file = 'pipeline_output'
    memory_info_dir = 'memory_info'
    def __init__(self, input_args, para):
        self.profile_mem_data = None
        self.input_args = input_args
        self.para = para
        self.peak_memory_usage = 0.0      # MB
        self.static_memory_usage = 0.0    # MB

    def write_mem_info_to_txt(self, config, mem_info):
        '''

        Args:
            config: [dp, tp, pp, ep, offset, recompute, num_layers]

        Returns:

        '''
        memory_dir = os.path.join(os.getcwd(), self.pipeline_output_file, self.memory_info_dir)
        os.makedirs(memory_dir, exist_ok=True)
        dense_layer_num = self.input_args.first_k_dense_replace
        moe_layer_num = self.input_args.num_layers - dense_layer_num

        dp, tp, pp, ep = config[:4]
        mbn = self.input_args.gbs // self.input_args.mbs // dp
        filename = (
            f'layers{dense_layer_num}_{moe_layer_num}_micro{mbn}_dp{dp}'
            f'_tp{tp}_pp{pp}_vp1'
            f'_ep{ep}.txt')
        memory_file_name = os.path.join(memory_dir, filename)
        mem_info.write_to_txt(memory_file_name)

    def set_static_mem(self, mem_info, profile_file):
        self.para.TOML_PATH = profile_file
        input_args = ParaForNd(self.para)
        dense_size, moe_size, vocab_size = compute_weight_and_optimizer_memory(input_args)
        mem_info.layer_mem012 = dense_size
        mem_info.layer_mem = moe_size
        mem_info.static_mem0 = vocab_size
        mem_info.static_mem = 0
        mem_info.lm_head_mem = 1.5 * moe_size
        return input_args

    def cal_dynamic_mem(self, mem_info, cur_input_args):
        dynamic_mem_total = self.peak_memory_usage - self.static_memory_usage
        if cur_input_args.pp == 1:
            dynamic_mem_layer = dynamic_mem_total // cur_input_args.num_layers
            mem_info.act_mem0 = dynamic_mem_layer
            mem_info.act_mem12 = dynamic_mem_layer
            mem_info.act_mem = dynamic_mem_layer



    def manage_mem_infos(self, need_cumlate=True):
        """
        Docstring for manage_mem_infos
        
        :param self: Description
        :param need_cumlate: Description
        """
        sorted_times = sorted(self.real_memory_changes.keys())
        cumulative_mem = 0
        times = []
        memory_usage = []
        for time in sorted_times:
            if need_cumlate:
                cumulative_mem += self.real_memory_changes[time]
                times.append(time)
                memory_usage.append(cumulative_mem)
            else:
                times.append(time)
                memory_usage.append(self.real_memory_changes[time])
        max_index = memory_usage.index(max(memory_usage))
        peak_time = times[max_index]
        peak_memory_usage = memory_usage[max_index]
        logger.info(f'Peak memory usage: {peak_memory_usage}, peak time: {peak_time}')
        self.peak_memory_usage = peak_memory_usage / 1024

    def parse_memory_block(self, filename):
        """
        Docstring for parse_memory_block
        
        :param self: Description
        :param filename: Description
        """
        real_memory_changes = defaultdict(int)
        static_memory = 0
        flag = False
        with open(filename, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            # real memory
            for row in reader:
                alloc_size = row['Size(KB)']
                if not alloc_size:
                    logger.debug('Size(KB) is none')
                    continue
                size = float(alloc_size)

                alloc_time = row['Allocation Time(us)']
                if not alloc_time:
                    logger.debug(f'Allocation Time is none size is {size}')
                    continue
                start = int(alloc_time.strip().split('.')[0])

                release_time = row['Active Release Time(us)']
                if not release_time:
                    logger.debug(f'Active Release Time is none, size is {size}')
                    continue
                end = int(release_time.strip().split('.')[0])

                real_memory_changes[start] += size  # 在start时增加内存
                real_memory_changes[end - 1] += 0  # for plt
                real_memory_changes[end] -= size  # 在end时减少内存

                if flag is False:
                    static_memory = int(row['Allocation Total Allocated(MB)'].strip().split('.')[0])

                    flag = True
                    real_memory_changes[start] += (static_memory * 1024)

        print(f'static mem is {static_memory} M')
        self.static_memory_usage = static_memory
        self.real_memory_changes = real_memory_changes
        self.manage_mem_infos()

    def parse_memory_file(self, profile_dir):
        """
        Docstring for parse_memory_file
        
        :param self: Description
        :param profile_dir: Description
        """
        # step1. parse operator_memory.csv
        root_path = Path(profile_dir)
        if not root_path.is_dir():
            logger.info(f"profile_result_dir:{profile_dir} not exist")
            return
        folders = [f for f in root_path.iterdir() if f.is_dir()]
        folders.sort()
        for rank_folder in folders:
            logger.info(f"parsing: {rank_folder}")
            path = find_file_by_name(rank_folder, 'operator_memory.csv')
            if not path:
                logger.error(f"No operator memory csv file in profile_dir {profile_dir} rank_dir {rank_folder}")
                continue
            self.parse_memory_block(path)

    def parse_mem(self, profile_dir, profile_file):
        '''

        Args:
            profile_dir: profile结果的路径

        Returns:

        '''
        mem_info = MemoryInfo()
        self.parse_memory_file(profile_dir)
        cur_input_args = self.set_static_mem(mem_info, profile_file)
        self.cal_dynamic_mem(mem_info, cur_input_args)
        # file mem_info
        return mem_info

    def mem_parser(self, profile_configs):
        '''

        Args:
            profile_configs: [dp, tp, pp, ep, offset, recompute, num_layers, profile_result_dir]

        Returns:

        '''
        for config in profile_configs:
            profile_dir = config[-1]
            profile_file = config[-2]
            mem_info = self.parse_mem(profile_dir, profile_file)
            self.write_mem_info_to_txt(config, mem_info)
