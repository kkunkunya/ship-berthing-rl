# 贡献指南

感谢您对 Ship Berthing RL 的关注！我们欢迎各种形式的贡献。

## 如何贡献

### 报告 Bug
- 使用 [Issue 模板](../../issues/new?template=bug_report.md) 提交 Bug 报告
- 描述清楚复现步骤、预期行为和实际行为
- 附上相关日志或截图

### 提出新功能
- 使用 [Feature Request 模板](../../issues/new?template=feature_request.md)
- 说明功能的使用场景和预期效果

### 提交代码
1. Fork 本仓库
2. 创建功能分支: `git checkout -b feature/your-feature`
3. 提交更改: `git commit -m "Add: your feature description"`
4. 推送分支: `git push origin feature/your-feature`
5. 创建 Pull Request

## 开发环境设置

```bash
# 克隆仓库
git clone https://github.com/kkunkunya/ship-berthing-rl.git
cd ship-berthing-rl

# 使用 uv 安装依赖
uv sync

# 运行训练
uv run python -m src.train.train
```

## Commit 消息规范
- `Add:` 新功能
- `Fix:` Bug 修复
- `Update:` 功能增强
- `Refactor:` 代码重构
- `Docs:` 文档更新

## 许可证
提交贡献即表示您同意将代码以 MIT 许可证发布。
