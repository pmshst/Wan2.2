for /L %%i in (1, 1, 18) do (
    python generate_local.py --task i2v-A14B --size "1280*720" --image=./last_frame.png --ckpt_dir ./Wan2.2-I2V-A14B --prompt "film grade professional quality"
)