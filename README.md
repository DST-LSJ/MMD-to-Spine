# MMD 转 Spine

将常见MMD人物动作VMD转换成现有Spine 4.3角色骨骼动画的Python工具，附Spine测试工程。

## 演示视频

[点击观看：MMD 转 Spine 效果演示（哔哩哔哩）](https://www.bilibili.com/video/BV1Lkh467E6h)

## 目录

```text
MMD转Spine/           Python程序和运行资源
  main.py            唯一运行入口
  Texture2D/         转换输出使用的贴图
  资源/通用骨架.json  可配置源骨架
  文档/              核心模块、目标模板和配置
spine/               Spine测试工程、图集和测试贴图
上传GitHub步骤.md    首次提交和后续更新操作说明
开发记事.md          本目录整理与后续开发记录
```

两份Texture2D分别服务于程序输出和Spine测试工程，保持现有相对位置，不合并或删除。

## 运行

需要Python 3.10或以上；此前验证环境为Python 3.14。仅使用标准库，无需安装第三方依赖。目标编辑器为Spine 4.3。

进入 `MMD转Spine` 文件夹，使用随附的 `极乐净土动作数据.vmd`，或把自己的VMD放在main.py旁边，然后运行：

```text
python main.py
```

或在仓库根目录运行：

```text
python MMD转Spine/main.py --folder MMD转Spine
```

默认从0秒开始、最多300秒，可修改main.py顶部参数。超过动作末尾自动截断。支持批量、防重复、进度条、完成后任意键退出或10秒自动退出。没有BAT入口。

生成JSON中的图片路径为 `Texture2D/`，以JSON所在目录为基准；复制输出时一起复制贴图目录。完整用法见 [使用说明](MMD转Spine/使用说明.md)。

## 测试

打开 `spine/极乐净土.spine` 进行编辑器检查，或将程序输出完整导入Spine 4.3。测试资源说明见 [spine/README.md](spine/README.md)。

代码采用通用源骨架与解析足IK，目标腿部采用三维腿平面展开、固定骨长及固定Negative双骨IK；不依赖任何人物PMX。通用骨架不是所有人物模型的精确复现，生成动作需要视觉验收。转身时左右肢体对调的观感问题暂缓处理，见 [待处理问题](MMD转Spine/文档/待处理问题.md)。

## 哪些文件进入Git

- 提交：测试动作 `MMD转Spine/极乐净土动作数据.vmd`、Python代码、运行配置、目标模板、程序贴图、Spine测试工程及测试图集/贴图、说明文档。
- 忽略：除上述测试动作外的本地VMD、生成的 `*_spine.json`、转换缓存、诊断报告、Python缓存、ZIP与旧打包校验清单。
- 忽略不会删除本地文件；首次克隆后即可使用随附VMD测试，也可自行添加其他人物动作。

## 项目信息与素材

原说明署名：美神自由梦；人物素材标注：pcr霞。原说明见 [说明.txt](MMD转Spine/说明.txt)，相关专栏链接保留在原说明中。

当前未新增开源LICENSE，也未为第三方贴图、动作或Spine工程授予新许可。仓库可见性不等于素材使用许可；后续如添加代码许可证，应单独说明其适用范围。
