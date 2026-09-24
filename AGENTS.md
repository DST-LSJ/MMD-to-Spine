# 本仓库开发入口

- 仓库根目录为MMD转Spine脚本；MMD转Spine/是程序，spine/是用户视觉测试工程。
- 以本目录实际文件及开发记事.md为准。先读README.md、MMD转Spine/使用说明.md和开发记事.md，再按任务核对代码。不要从旧工程或旧聊天覆盖当前实现。
- 正式入口MMD转Spine/main.py，仅Python，无BAT。通用源骨架位于资源/通用骨架.json，不依赖人物PMX；骨长固定、模板设置姿势保持，身体/头部允许符号镜像。
- 导出skeleton.images必须为Texture2D/相对路径。保留批量扫描、防重复、实际处理进度及交互控制台10秒退出规则。
- 转身镜像导致左右肢体对调的观感问题按用户要求暂缓，后续先做独立测试版，不直接修改正式效果。
- 两个Texture2D目录分别服务程序与测试工程，不擅自合并。提交核心代码、必要资源及用户指定的测试动作MMD转Spine/极乐净土动作数据.vmd；其他本地VMD、缓存、转换输出和诊断由.gitignore排除。
- 修改后更新开发记事.md，并区分代码/数值检查与Spine视觉验收。未得到明确上传指令时不自动创建远程仓库或推送。
- 本机可使用cmd.exe、login=false，避免已知PowerShell CET启动问题。路径中包含方括号，PowerShell文件操作必须使用-LiteralPath。
