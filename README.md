# Compliant Humanoid

使用 uv 管理环境。训练代码位于 src/compliant_humanoid，G1 MJCF、URDF、网格和演示动作位于 data。请在仓库根目录执行命令。

## NVIDIA GPU 训练

~~~powershell
uv sync --locked --extra cuda
uv run --locked --extra cuda compliant-humanoid --iterations 10 --output policy.pt
~~~

默认使用 CUDA，Genesis 物理仿真和 PyTorch 策略更新共用显卡。参考轨迹插值、MuJoCo 运动学和 SciPy IK 在 CPU 上计算。当前训练使用单环境。CPU 验证可显式运行：

~~~powershell
uv run compliant-humanoid --device cpu --iterations 1
uv run python -m unittest discover -s tests
~~~

默认动作是仓库自带的 30 Hz 合成屈膝轨迹。tools/generate_demo.py 可重新生成它。本机也可传入 --clip data/lafan1_g1/walk4_subject1.csv。CSV 每行依次为根位置、xyzw 根四元数和按 G1 MJCF 排列的 29 个关节角。训练重采样为 50 Hz，物理频率为 200 Hz。

## 实现范围

策略包含力、平衡和未来参考编码器、29 维动作和非对称价值网络。单轮训练包含左腕扰动、固定根姿态的 IK、关节跟踪和动作平滑奖励、GAE、PPO 和辅助估计损失。完整奖励、域随机化、长回合和论文任务评测仍未实现。

G1 资源来自 Unitree Robotics，许可证位于 data/g1_description/LICENSE。walk4_subject1.csv 是 Ubisoft La Forge LAFAN1 的 G1 重定向片段，按源项目说明仅保存在本地。仓库中的 knee_bend.csv 由生成脚本独立生成。
