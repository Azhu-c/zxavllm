# 算子构建依赖源码包

本目录保存正常 ACLNN 构建使用的四个固定版本源码包，供公司服务器无法访问下载地址时使用。文件来自原有 CMake 脚本的公开下载地址，未重新压缩；每个包保留上游许可和版权文件。

| 依赖 | 版本 | 原始下载地址 |
| --- | --- | --- |
| makeself | 2.5.0-patch1.0 | [CANN makeself 源码包](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) |
| nlohmann JSON | 3.11.3 | [CANN JSON 头文件包](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| Abseil | 20230802.1 | [CANN Abseil 源码包](https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz) |
| Protobuf | 25.1 | [CANN Protobuf 源码包](https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz) |

`SHA256SUMS` 固定这些包的校验值；makeself 的值与原构建脚本中已有的 SHA256 完全一致。`offline_archive.cmake` 在配置时校验本地文件，随后由 FetchContent/ExternalProject 再按同一 SHA256 验证和解压。发现损坏的本地包时停止构建；文件缺失时保留原下载地址或原有的 CANN 包缓存回退。

构建脚本会自动选择这些包，更新完整源码后直接重新执行源码安装即可。请保留本目录的四个压缩包、`SHA256SUMS` 和上一级的 CMake 文件；不需要手动将 makeself 解压到某个构建目录，也不需要关闭 TLS 校验或修改服务器的 CMake 版本。

这些是构建工具及库的源码包，不包含 CANN SDK、NPU 驱动、Python 依赖或已编译算子。CATLASS 仍使用仓库根目录的固定版本头文件包。可选测试使用的 GTest 等依赖不在这四个包中。

回归测试使用真实 CMake 配置和构建依赖步骤，包括 makeself 权限修复、JSON 头文件安装、Abseil/Protobuf 的现有 CANN 补丁，以及无网络环境和错误校验值。测试不依赖 NPU；完整算子编译和运行仍在服务器验证。
