# 修复源码安装时 catlass 缺失

已根据服务器日志定位：`dependency catlass is missing` 后，构建脚本执行 `git submodule update`；传输后的 `/workspace/dyntralb` 没有可用 Git 仓库信息，因此 `fetch failed`，导致 editable wheel 构建失败。

本分支现已在构建脚本中接入离线恢复：缺少 CATLASS 时，先校验仓库随附压缩包的 SHA256，再自动解压。通过 Git 克隆或下载完整分支源码后，在原服务容器或虚拟环境执行：

```bash
cd /workspace/dyntralb
python -m pip install -e . --no-deps --no-build-isolation
```

请保留源码根目录的 `.gitmodules`、`catlass-41bf90da-headers.tar.gz` 和 `catlass-41bf90da-headers.tar.gz.sha256`。离线包缺失时，有 Git 元数据的 checkout 会回退到原来的子模块下载；没有 Git 元数据则会报出需要补齐的文件。压缩包校验失败时停止构建，不会静默切换依赖版本。不会修改用户的全局 Git 配置。

若服务器还在使用旧构建脚本，只需将 `catlass-41bf90da-headers.tar.gz` 上传到服务器 `/workspace/dyntralb/`，手动解压后再安装：

```bash
cd /workspace/dyntralb
tar -xzf catlass-41bf90da-headers.tar.gz
ls csrc/third_party/catlass/include/catlass/catlass.hpp
python -m pip install -e . --no-deps --no-build-isolation
```

包内路径已经包含 `csrc/third_party/catlass/`，请直接在源码根目录解压，不要额外使用 `--strip-components`。`ls` 应输出该头文件路径；若失败，先确认解压位置再安装。保持原来的 CANN、SOC_VERSION、vLLM 和 TorchNPU 设置。此操作不要求重新初始化整个源码目录的 Git，也不需要设置 `GIT_DISCOVERY_ACROSS_FILESYSTEM`。

## 版本和范围

- 来源：[CATLASS 官方仓库](https://gitcode.com/cann/catlass.git)
- 固定提交： `41bf90da655bba3c66d0acd7e00abe33960ecfd6`，与此版 `.gitmodules` 和 submodule gitlink 一致。
- 包含全部 233 个 `include/` 文件、上游 LICENSE、README 和许可通知。
- 仅补全 CATLASS 构建头文件；不包含 CANN、编译器或已编译算子。CATLASS 自身单元测试的 googletest 子模块不包含在内，当前 vllm-ascend 构建不需要它。
- 已验证解压后所有头文件与固定提交逐个一致，并在无 `.git` 的临时目录测试自动恢复、重复运行、校验失败、损坏的压缩包和依赖缺失报错；另用本地 Git 仓库测试子模块下载回退，不依赖服务器联网。
- 未在 Ascend 服务器执行完整构建，后续其他构建错误需要按新的错误处理。

`listing git files failed` 和 `0.0.0` 可能仍出现在无 Git 信息的源码构建日志中；当前附件里的致命退出来自 catlass 的 submodule fetch，补头文件解决的是这一处退出。
