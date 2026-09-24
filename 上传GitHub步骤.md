# 第一次上传到GitHub

Git管理本地版本；GitHub保存远程仓库。以下默认使用GitHub；如使用Gitee，只需替换远程仓库地址并按其登录提示操作。

## 1. 创建空远程仓库

登录 https://github.com/new ，填写仓库名，例如MMD-to-Spine。先选择Private可供自己核对，确认后可再调整可见性。不要勾选自动添加README、.gitignore或License，因为本地已经有内容。创建完成后复制HTTPS仓库地址。

## 2. 打开正确的本地文件夹

在资源管理器进入同时包含MMD转Spine和spine的父目录，在地址栏输入cmd并回车。不要在两个子目录分别初始化仓库。

也可在cmd中运行：

```bat
cd /d "E:\公主连结\[教程]如何提取公主连结游戏动画素材\MMD转Spine脚本"
git status
```

整理时已初始化本地main分支，无需再运行git init。此时尚无提交。

## 3. 设置本仓库提交署名

先查看是否已有署名：

```text
git config user.name
git config user.email
```

没有值或希望另用署名时，替换示例文本后运行（仅影响当前仓库）：

```text
git config user.name "你的署名"
git config user.email "你的提交邮箱"
```

可以使用GitHub账户设置中Emails页面提供的noreply邮箱。不要把账户密码或访问令牌填成提交邮箱。

## 4. 暂存并检查

```text
git add .
git status
git diff --cached --stat
```

应看到两个目录的代码、模板、贴图、Spine工程、说明，以及测试动作MMD转Spine/极乐净土动作数据.vmd；不应看到.mmd_spine_cache.json、其他本地VMD、生成的*_spine.json或__pycache__。发现不希望上传的文件可先执行 `git rm --cached "文件路径"` 取消首次暂存，它保留本地文件，然后补充.gitignore。

## 5. 创建第一次本地提交

```text
git commit -m "Initial import: MMD to Spine converter and test assets"
```

## 6. 关联远程仓库

当前已创建仓库DST-LSJ/MMD-to-Spine，使用以下地址：

```text
git remote add origin https://github.com/DST-LSJ/MMD-to-Spine.git
git remote -v
```

若提示origin已存在，先用git remote -v核对，不要重复添加。地址确需修改时使用 `git remote set-url origin 新地址`。

## 7. 上传

```text
git push -u origin main
```

按Git弹出的浏览器/凭据管理器完成GitHub登录。不要把密码或令牌写入命令、README或仓库文件。如果提示远程已有提交，先停下核对，不要直接强制推送。

完成后刷新GitHub仓库，确认README、MMD转Spine和spine三个主要入口可见。

## 以后更新

```text
git status
git add .
git diff --cached --stat
git commit -m "说明这次修改"
git push
```

每次先检查实际暂存内容。没有文件变化时不需要重复提交。其他电脑也改过远程仓库时，在本地修改前先同步；有未提交工作时先处理当前工作，不直接覆盖。

官方流程：[将本地代码添加到GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)。
