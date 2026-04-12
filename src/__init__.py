from gymnasium.envs.registration import register

register(
    id="gymnasium_env/GridWorld-v0", # namespace/name-version
    entry_point="src.environments:GridWorldEnv" # Corresponds to path: folder/folder/folder_in_which_file_exists:ClassName
)