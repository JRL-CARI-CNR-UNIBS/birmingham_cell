from gymnasium.envs.registration import register

register(
    id="BirminghamCellEnv-v0",
    entry_point="birmingham_envs.envs:BirminghamCellEnv",
)
register(
    id="ConnectionEnv-v0",
    entry_point="birmingham_envs.envs:ConnectionEnv",
)

register(
    id="FakeEnv-v0",
    entry_point="birmingham_envs.envs:FakeEnv",
)

register(
    id="EasyEnv-v0",
    entry_point="birmingham_envs.envs:EasyEnv",
)

register(
    id="RealisticFakeEnv-v0",
    entry_point="birmingham_envs.envs:RealisticFakeEnv",
)

register(
    id="RandomRealFakeEnv-v0",
    entry_point="birmingham_envs.envs:RandomRealFakeEnv",
)

register(
    id="GenericRealFakeEnv-v0",
    entry_point="birmingham_envs.envs:GenericRealFakeEnv",
)

register(
    id="RealisticFakeEnv2-v0",
    entry_point="birmingham_envs.envs:RealisticFakeEnv2",
)

register(
    id="RealisticForceFakeEnv-v0",
    entry_point="birmingham_envs.envs:RealisticForceFakeEnv",
)

register(
    id="GenericRealForceFakeEnv-v0",
    entry_point="birmingham_envs.envs:GenericRealForceFakeEnv",
)

register(
    id="RealHistoryFakeEnv-v0",
    entry_point="birmingham_envs.envs:RealHistoryFakeEnv",
)

register(
    id="RealPosFeedbackFakeEnv-v0",
    entry_point="birmingham_envs.envs:RealPosFeedbackFakeEnv",
)

register(
    id="GeneralEnv-v0",
    entry_point="birmingham_envs.envs:GeneralEnv",
)

register(
    id="ConnectionHistoryEnv-v0",
    entry_point="birmingham_envs.envs:ConnectionHistoryEnv",
)

register(
    id="TOFConnectionEnv-v0",
    entry_point="birmingham_envs.envs:TOFConnectionEnv",
)

register(
    id="StaticRealFakeEnv-v0",
    entry_point="birmingham_envs.envs:StaticRealFakeEnv",
)

register(
    id="StaticConnectionEnv-v0",
    entry_point="birmingham_envs.envs:StaticConnectionEnv",
)
