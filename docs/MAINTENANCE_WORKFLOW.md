# v1.x 维护工作流程

## Bug修复流程

### 1. 创建bug修复分支

```bash
git checkout v1.x-maintenance
git pull origin v1.x-maintenance
git checkout -b hotfix/v1.0.x-<bug-description>
```

### 2. 修复bug

- 修复代码
- 添加测试（如适用）
- 更新CHANGELOG.md

### 3. 提交和测试

```bash
git add .
git commit -m "fix: <bug description> (v1.0.x)"
git push origin hotfix/v1.0.x-<bug-description>
```

### 4. 创建Pull Request

- 从`hotfix/v1.0.x-*`到`v1.x-maintenance`
- 描述bug和修复方案
- 等待代码审查

### 5. 合并和发布

```bash
# 合并到v1.x-maintenance
git checkout v1.x-maintenance
git merge hotfix/v1.0.x-<bug-description>
git push origin v1.x-maintenance

# 创建版本tag
git tag -a v1.0.1 -m "Bug fix release"
git push origin v1.0.1

# 同步到archive分支（可选）
git checkout archive/gui-version
git merge v1.x-maintenance
git push origin archive/gui-version
```

## 版本号规则

- **主版本号**：1（保持不变）
- **次版本号**：0（保持不变）
- **修订号**：递增（1, 2, 3...）

示例：v1.0.0 → v1.0.1 → v1.0.2