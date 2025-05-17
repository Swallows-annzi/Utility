from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import configparser

app = FastAPI()

origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    text: str


from pydantic import BaseModel

class TrainingRequest(BaseModel):
    Epochs: int
    Test: float
    LearningRate: float
    Modes: str
    UserName: str
    FileName: str

# 接收命令的API
@app.post("/api/upDict")
async def up_dict(request: TrainingRequest):
    try:
        Epochs = request.Epochs
        Test = request.Test
        LearningRate = request.LearningRate
        Modes = request.Modes
        UserName = request.UserName
        FileName = request.FileName.split('.')[0]

        command = f'python backend/Main.py --UserName {UserName} --FileIn backend/Input/{FileName} --Test {Test} --LearningRate {LearningRate} --Mode {Modes} --Epochs {Epochs}'
        
        try:
            subprocess.Popen(command, shell=True)
            return {"message": "训练已启动，将在后台运行"}
        except Exception as e:
            print('命令执行出错:', str(e))
            return {"message": f"训练启动出错: {str(e)}"}, 422

    except Exception as e:
        return {"message": f"处理请求时出错: {str(e)}"}, 422

# 发送用户记录的API
@app.get("/api/getFiles")
async def get_files(username: str):
    try:
        import os
        output_dir = os.path.join("backend", "Output", username)
        print('尝试访问的目录:', output_dir)
        if os.path.exists(output_dir):
            folders = [name for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, name))]
            print('找到的文件夹:', folders)
            return folders
        else:
            print('用户不存在:', output_dir)
            return []
    except Exception as e:
        print(f'获取用户文件夹时出现错误: {e}')
        return []

# 删除记录的api
from pydantic import BaseModel

class RemoveDataRequest(BaseModel):
    username: str
    fileout: str

@app.post("/api/removeData")
async  def remove_data(request: RemoveDataRequest):
    try:
        username = request.username
        fileout = request.fileout
        import shutil
        file_path = os.path.join("backend", "Output", username, fileout)
        if os.path.exists(file_path):
            shutil.rmtree(file_path)
            print(f"已删除文件夹：{file_path}")
            return {"message": "记录删除成功"}
        else:
            print(f"文件夹不存在：{file_path}")
            return {"message": "记录不存在", "status": 404}, 404
    except Exception as e:
        print(f'删除记录时出现错误: {e}')
        return {"message": f"删除记录时出错: {str(e)}", "status": 500}, 500

# 发送单项训练记录的API
@app.get("/api/getData")
async def get_data(username: str, fileout: str):
    try:
        config_path = os.path.join("backend", "Output", username, fileout, "Config.cfg")

        config = configparser.ConfigParser()
        config.read(config_path)

        if 'Parameters' in config:
            parameters = config['Parameters']
            data = {
                'State': 'Running',
                'Learningrate': parameters.get('learningrate'),
                'Mode': parameters.get('mode'),
                'Epochs': parameters.get('epochs'),
                'Test': parameters.get('test'),
                'Fileout': parameters.get('fileout'),
                'UserName': parameters.get('username')
            }
            cmc_json_path = os.path.join("backend", "Output", username, fileout, "cmc.json")
            if os.path.exists(cmc_json_path):
                import json
                with open(cmc_json_path, 'r', encoding='utf-8') as f:
                    cmc_data = json.load(f)
                data['State'] = 'Finish'
                models = ['cnn', 'rnn', 'mlp', 'lstm', 'transformer']
                for model in models:
                    if model in cmc_data:
                        data[model] = cmc_data[model]
            print('发送数据:', data)
            return data
        else:
            return []
    
    except Exception as e:
        print(f'获取数据时出现错误: {e}')

# 上传训练集的API
@app.post("/api/upFile")
async def up_file(file: UploadFile = File(...)):
    input_dir = "backend/Input"
    if not os.path.exists(input_dir):
        os.makedirs(input_dir)
    
    try:
        file_path = os.path.join(input_dir, file.filename)
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
    except Exception:
        return {"message": "上传文件时发生错误"}
    finally:
        await file.close()
    
    return {"message": f"文件 {file.filename} 上传成功", "filename": file.filename}