<template>
  <div class="home-container">
    <el-card class="main-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span>训练平台</span>
        </div>
      </template>
      <el-form :inline="true" class="username">
        <el-form-item label="用户名">
          <el-input 
            v-model="username" 
            placeholder="请输入用户名"
            :disabled="isLocked"
          ></el-input>
        </el-form-item>
        <el-form-item>
          <el-button 
            type="primary" 
            @click="toggleLock"
          >
            {{ isLocked ? '解锁' : '锁定' }}
          </el-button>
        </el-form-item>
        <el-form-item>
          <el-button 
            type="success" 
            @click="fetchRecords"
            :disabled="!isLocked || username.value"
          >
            刷新数据
          </el-button>
        </el-form-item>
        <el-form-item>
          <el-button 
            type="info" 
            @click="toggleSort"
            :disabled="!isLocked || username.value"
          >
            {{ isAscending ? '正序显示' : '倒序显示' }}
          </el-button>
        </el-form-item>
        <el-form-item>
          <el-button 
            type="warning" 
            @click="showHelloPage"
            :disabled="!isLocked || username.value"
          >
            创建训练
          </el-button>
        </el-form-item>
      </el-form>
      <el-table :data="sortedRecords" style="width: 100%;" stripe fit height="580">
        <el-table-column prop="time" label="时间" width="180"></el-table-column>
        <el-table-column label="操作">
          <template #default="scope">
            <el-button 
              type="primary" 
              size="small" 
              class="beautified-button"
              @click="fetchAndPrintData(scope.row.time)"
            >
              信息
            </el-button>
            <el-button 
              type="danger" 
              size="small" 
              class="beautified-button"
              @click="deleteRecord(scope.row.time)"
            >
              删除
            </el-button>
          </template>
        </el-table-column>
      </el-table>
      <el-dialog v-model="dialogVisible" title="创建训练">
        <el-form-item label="用户名">
          <span>{{ username }}</span>
        </el-form-item>
        <el-form-item label="训练轮数">
          <el-input 
            v-model="Epochs" 
            type="number" 
            :min="1" 
            placeholder="请输入训练轮数"
          ></el-input>
        </el-form-item>
        <el-form-item label="测试集分组比例">
          <el-input 
            v-model="Test" 
            type="number" 
            :min="0.01" 
            placeholder="请输入分组比例"
          ></el-input>
        </el-form-item>
        <el-form-item label="学习率">
          <el-input 
            v-model="LearningRate" 
            type="number" 
            :min="0.0001" 
            placeholder="请输入学习率"
          ></el-input>
        </el-form-item>
        <el-form-item>
          <el-button 
            :type="isCNNSelected ? 'primary' : 'info'" 
            @click="toggleModel('CNN')"
          >
            CNN
          </el-button>
          <el-button 
            :type="isRNNSelected ? 'primary' : 'info'" 
            @click="toggleModel('RNN')"
          >
            RNN
          </el-button>
          <el-button 
            :type="isMLPSelected ? 'primary' : 'info'" 
            @click="toggleModel('MLP')"
          >
            MLP
          </el-button>
          <el-button 
            :type="isLSTMSelected ? 'primary' : 'info'" 
            @click="toggleModel('LSTM')"
          >
            LSTM
          </el-button>
          <el-button 
            :type="isTransformerSelected ? 'primary' : 'info'" 
            @click="toggleModel('Transformer')"
          >
            Transformer
          </el-button>
        </el-form-item>
        <el-form-item label="文件上传">
          <el-upload
            action="http://localhost:8001/api/upFile"
            :on-success="handleUploadSuccess"
            :on-error="handleUploadError"
            :before-upload="beforeUpload"
            :limit="1"
            :on-exceed="handleExceed"
            :on-change="handleChange"
            list-type="text"
            :auto-upload="true"
          >
            <el-button type="primary">点击上传</el-button>
          </el-upload>
        </el-form-item>
        <template #footer>
          <span class="dialog-footer">
            <el-button @click="handleCreate">创建</el-button>
            <el-button @click="dialogVisible = false">关闭</el-button>
          </span>
        </template>
      </el-dialog>
      <el-dialog v-model="dataDialogVisible" title="训练数据" width="60%">
        <div class="data-display">
          <p>用户名: {{ fetchedData.UserName }}</p>
          <p>创建时间: {{ fetchedData.Fileout }}</p>
          <p>状态: {{ getStatusText(fetchedData.State) }}</p>
          <p>模型: {{ fetchedData.Mode }}</p>
          <p>训练轮数: {{ fetchedData.Epochs }}</p>
          <p>测试集分组比例: {{ fetchedData.Test }}</p>
          <p>学习率: {{ fetchedData.Learningrate }}</p>
          <div v-if="fetchedData.State === 'Finish'">
            <h3>模型详细数据</h3>
            <el-table :data="getExistingModelsData()" style="width: 95%;" stripe>
              <el-table-column prop="model" label="模型" width="130"></el-table-column>
              <el-table-column prop="accuracy" label="Accuracy"></el-table-column>
              <el-table-column prop="precision" label="Precision"></el-table-column>
              <el-table-column prop="recall" label="Recall"></el-table-column>
              <el-table-column prop="f1Score" label="F1 Score"></el-table-column>
              <el-table-column prop="mcc" label="MCC"></el-table-column>
              <el-table-column prop="auc" label="AUC"></el-table-column>
            </el-table>
          </div>
        </div>
        <!-- <pre>{{ JSON.stringify(fetchedData, null, 2) }}</pre> -->
      </el-dialog>
    </el-card>
  </div>
</template>

<script setup>
import { ElMessage, ElCard, ElForm, ElFormItem, ElInput, ElTable, ElTableColumn, ElButton, ElDialog, ElMessageBox } from 'element-plus';
import { ref, computed } from 'vue';
import axios from 'axios';

const username = ref('');
const records = ref([]);
const isLocked = ref(false);
const isAscending = ref(true); 

const dialogVisible = ref(false);
const Epochs = ref(10);
const Test = ref(0.2);
const LearningRate = ref(0.001);
const FileName = ref('');

const isCNNSelected = ref(false);
const isRNNSelected = ref(false);
const isMLPSelected = ref(false);
const isLSTMSelected = ref(false);
const isTransformerSelected = ref(false);


// 处理创建按钮点击事件
const handleCreate = async () => {
  if (!FileName.value) {
    ElMessage.warning('请先上传文件');
    return;
  }
  if (!getSelectedModes()) {
    ElMessage.warning('请选择至少一个模型');
    return;
  }

  const data = {
    Epochs: Number(Epochs.value),
    Test: Number(Test.value),
    LearningRate: Number(LearningRate.value),
    Modes: getSelectedModes(),
    UserName: username.value,
    FileName: FileName.value
  };

  console.log('即将发送的数据:', data);
  try {
    dialogVisible.value = false;
    const url = `http://localhost:8001/api/upDict`;
    const response = await axios.post(url, data, {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    ElMessage.success('创建训练成功，模型正在训练中');
  } catch (error) {
    if (error.response && error.response.status === 422) {
      console.error('后端错误信息:', error.response.data);
      ElMessage.error('请求数据格式不正确，请检查输入内容');
    } else {
      ElMessage.error('创建训练失败，请稍后重试');
    }
    console.error('创建训练失败:', error);
  }
};

// 获取记录数据
const fetchRecords = async () => {
  if (!username.value) {
    ElMessage.warning('请输入用户名');
    return;
  }
  try {
    const url = `http://localhost:8001/api/getFiles?username=${username.value}`;
    console.log('请求URL:', url); 
    console.log('用户名:', username.value); 
    const response = await axios.get(url);
    console.log('API 返回数据:', response.data); 
    if (Array.isArray(response.data)) {
      if (response.data.length === 0) {
        ElMessage.info('无记录');
      }
      records.value = response.data.map((timeStr, index) => ({
        time: timeStr, 
      }));
    } else {
      console.error('返回的数据不是数组，实际数据类型:', typeof response.data, '数据内容:', response.data);
      ElMessage.error('获取的记录数据格式不正确，请稍后重试');
    }
  } catch (error) {
    if (error.response) {
      ElMessage.error(`获取记录失败，状态码: ${error.response.status}，错误信息: ${error.response.data.message || '未知错误'}，请稍后重试`);
    } else if (error.request) {
      ElMessage.error('获取记录失败，未收到服务器响应，请检查网络连接，稍后重试');
    } else {
      ElMessage.error(`获取记录失败，错误信息: ${error.message}，请稍后重试`);
    }
    console.error('获取记录失败:', error);
  }
};

const toggleSort = () => {
  isAscending.value = !isAscending.value;
};

const toggleLock = () => {
  isLocked.value = !isLocked.value;
};

const sortedRecords = computed(() => {
  return [...records.value].sort((a, b) => {
    if (isAscending.value) {
      return a.time.localeCompare(b.time);
    } else {
      return b.time.localeCompare(a.time);
    }
  });
});

const showHelloPage = () => {
  dialogVisible.value = true;
};

const toggleModel = (model) => {
  switch (model) {
    case 'CNN':
      isCNNSelected.value = !isCNNSelected.value;
      break;
    case 'RNN':
      isRNNSelected.value = !isRNNSelected.value;
      break;
    case 'MLP':
      isMLPSelected.value = !isMLPSelected.value;
      break;
    case 'LSTM':
      isLSTMSelected.value = !isLSTMSelected.value;
      break;
    case 'Transformer':
      isTransformerSelected.value = !isTransformerSelected.value;
      break;
  }
};

const getSelectedModes = () => {
  const selectedModes = [];
  if (isCNNSelected.value) selectedModes.push('cnn');
  if (isRNNSelected.value) selectedModes.push('rnn');
  if (isMLPSelected.value) selectedModes.push('mlp');
  if (isLSTMSelected.value) selectedModes.push('lstm');
  if (isTransformerSelected.value) selectedModes.push('transformer');
  return selectedModes.join('+');
};

// 处理文件选择变化
const handleChange = (file) => {
  FileName.value = file.name;
};

// 处理文件上传成功
const handleUploadSuccess = (response, file, fileList) => {
  ElMessage.success('文件上传成功');
};

// 处理文件上传失败
const handleUploadError = (error, file, fileList) => {
  ElMessage.error('文件上传失败，请稍后重试');
  selectedFileName.value = '';
};

// 上传前检查文件类型
const beforeUpload = (file) => {
  const isFasta = file.name.endsWith('.fasta');

  if (!isFasta) {
    ElMessage.error('只能上传.fasta 文件');
    return false;
  }
  return true;
};

// 处理超过文件数量限制
const handleExceed = (files, fileList) => {
  ElMessage.warning('只能上传一个文件');
};

const dataDialogVisible = ref(false);
const fetchedData = ref({});

const fetchAndPrintData = async (fileout) => {
  if (!username.value) {
    ElMessage.warning('请输入用户名');
    return;
  }
  try {
    const url = `http://localhost:8001/api/getData?username=${username.value}&fileout=${fileout}`;
    const response = await axios.get(url);
    fetchedData.value = response.data;
    dataDialogVisible.value = true;
  } catch (error) {
    if (error.response) {
      ElMessage.error(`获取数据失败，状态码: ${error.response.status}，错误信息: ${error.response.data.message || '未知错误'}，请稍后重试`);
    } else if (error.request) {
      ElMessage.error('获取数据失败，未收到服务器响应，请检查网络连接，稍后重试');
    } else {
      ElMessage.error(`获取数据失败，错误信息: ${error.message}，请稍后重试`);
    }
    console.error('获取数据失败:', error);
  }
};

// 根据状态返回对应的文字描述
const getStatusText = (state) => {
  if (state === 'Running') {
    return '训练中';
  } else if (state === 'Finish') {
    return '训练完成';
  } else {
    return state;
  }
};

// 计算指标的方法
const calculateMetrics = (tp, fp, tn, fn, aucD) => {
  const accuracy = (tp + tn) / (tp + fp + tn + fn);
  const precision = tp / (tp + fp);
  const recall = tp / (tp + fn);
  const f1Score = 2 * (precision * recall) / (precision + recall);
  const mccNumerator = (tp * tn - fp * fn);
  const mccDenominator = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn));
  const mcc = mccDenominator === 0 ? 0 : mccNumerator / mccDenominator;
  const auc = aucD || 0; 

  return {
    accuracy: accuracy.toFixed(4),
    precision: precision.toFixed(4),
    recall: recall.toFixed(4),
    f1Score: f1Score.toFixed(4),
    mcc: mcc.toFixed(4),
    auc: auc.toFixed(4)
  };
};

// 获取存在数据的模型数据
const getExistingModelsData = () => {
  const models = ['cnn', 'rnn', 'mlp', 'lstm', 'transformer'];
  const existingModelsData = [];

  models.forEach(model => {
    if (fetchedData.value[model]) {
      const { tp, fp, tn, fn, auc } = fetchedData.value[model];
      const metrics = calculateMetrics(tp, fp, tn, fn, auc);
      existingModelsData.push({
        model: model.toUpperCase(),
        ...metrics
      });
    }
  });

  return existingModelsData;
};

const deleteRecord = async (fileout) => {
  if (!username.value) {
    ElMessage.warning('请输入用户名');
    return;
  }
  try {
    const confirmDelete = await ElMessageBox.confirm(
      '确定要删除这条记录吗？',
      '提示',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning',
      }
    );
    if (confirmDelete) {
      const url = `http://localhost:8001/api/removeData`;
      const data = {
        username: username.value,
        fileout: fileout
      };
      const response = await axios.post(url, data);
      ElMessage.success(response.data.message);
      await fetchRecords();
    }
  } catch (error) {
    if (error.type === ElMessageBox.ERROR_CANCEL_ACTION) {
      return;
    }
    if (error.response) {
      ElMessage.error(`删除记录失败，状态码: ${error.response.status}，错误信息: ${error.response.data.message || '未知错误'}，请稍后重试`);
    } else if (error.request) {
      ElMessage.error('删除记录失败，未收到服务器响应，请检查网络连接，稍后重试');
    } else {
      ElMessage.error(`删除记录失败，错误信息: ${error.message}，请稍后重试`);
    }
    console.error('删除记录失败:', error);
  }
};
</script>

<style scoped>
.home-container {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
  height: 100vh;
}

.main-card {
  border-radius: 8px;
  box-shadow: 0 2px 12px 0 rgba(0, 0, 0, 0.1);
  height: 100%;
  display: flex;
  flex-direction: column;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 18px;
  font-weight: 600;
  padding: 15px 20px;
}

.demo-form-inline {
  margin-top: 20px;
  padding: 0 20px;
}

.el-table {
  flex: 1;
  margin: 20px;
}

.beautified-button {
  background-color: #409EFF;
  color: white;
  border-radius: 4px;
  transition: all 0.3s;
}

.beautified-button:hover {
  background-color: #66B1FF;
  transform: scale(1.05);
}

.data-display {
  padding: 20px;
  border-radius: 8px;
  background-color: #f9f9f9;
  border: 1px solid #eaeaea;
}

.data-display p {
  margin: 10px 0;
  font-size: 14px;
  color: #333;
}
</style>
