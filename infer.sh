# rm -r results
CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p225/p225_001.wav --ref_spk p225
# CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p227/p227_003.wav --ref_spk p227
CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p231/p231_007.wav --ref_spk p231
CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p232/p232_005.wav --ref_spk p232
CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p230/p230_007.wav --ref_spk p230
CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p236/p236_011.wav --ref_spk p236
CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p239/p239_014.wav --ref_spk p239
CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p243/p243_015.wav --ref_spk p243
CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p233/p233_016.wav --ref_spk p233
CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p228/p228_018.wav --ref_spk p228
CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p254/p254_008.wav --ref_spk p254
CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p256/p256_020.wav --ref_spk p256
CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p258/p258_021.wav --ref_spk p258
CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p299/p299_034.wav --ref_spk p299
CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p300/p300_035.wav --ref_spk p300

# CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p277/p277_163.wav --ref_spk p277
# CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p281/p281_175.wav --ref_spk p281
# CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p315/p315_019.wav --ref_spk p315
# CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p339/p339_046.wav --ref_spk p339
# CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p341/p341_123.wav --ref_spk p341
# CUDA_VISIBLE_DEVICES=1 python synthesize.py --checkpoint_path exp_ch16_ker7/ckpt/checkpoint_800000.pth.tar --ref_audio /home/hcy71/VCTK-Corpus/wav48/p374/p374_017.wav --ref_spk p374

cd hifi-gan

python inference_e2e.py --checkpoint_file cp_hifigan/g_02505000
