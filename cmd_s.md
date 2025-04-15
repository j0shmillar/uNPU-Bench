## model
### vela

```
vela --accelerator-config ethos-u55-64 --recursion-limit 2000 --optimise {Size|Performance} ./model/{model}/quant/{model}_full_integer_quant.tflite --output-dir ./model/{model}/vela_{size|perf} 
```

### eiq

```
/opt/nxp/eIQ_Toolkit_v1.13.1/bin/neutron-converter/MCU_SDK_2.16.000/neutron-converter --input {model}_full_integer_quant.tflite --output {model}_full_integer_quant_eiq.tflite
```

### mlir / cvi

```
docker run --privileged --name myname1234 -v $PWD:/workspace -it sophgo/tpuc_dev:latest
source tpu-mlir/envsetup.sh
model_transform.py --model_name {model} --model_def model/{model}/{model}.onnx --mlir model/{model}/{model}.mlir
cd model/{model}
run_calibration.py {model}.mlir --dataset dataset --input_num 1 -o calibration_table
model_deploy.py --mlir {model}.mlir --quant_input --quant_output --quantize INT8 --calibration_table calibration_table --chip cv180x --test_reference {model}_top_f32_all_weight.npz --tolerance 0.85,0.45 --model {model}.cvimodel
```

## npu
### luckfox
on linux machine, do:
```
cd npu/luckfox/{model}
./build.sh
```
then
```
adb push luckfox_pico_{model} root
adb shell
cd root/luckfox_pico_{model}_demo
./luckfox_pico_{model} model/{model}.rknn
```

### milkv

```
cd host-tools
export PATH=$PATH:$(pwd)/gcc/riscv64-linux-musl-x86_64/bin
cd ../duo-examples
source envsetup.sh
cd ../cvitek-tdl-sdk-sg200x/sample (change 2 npu/milkv or npu/cv1800b)
./compile_sample.sh
scp -O sample_{model} root@192.168.42.1:/root/
scp -O ../../model/{model}/{model}.cvimodel root@192.168.42.1:/root/
ssh root@192.168.42.1
cd root
./sample_{model} {model}.cvimodel
password = milkv
```

### himax

build (update `EPII_CM55M_APP_S/makefile`, line 133 set `APP_TYPE` for demos)

```
cd ~
wget https://developer.arm.com/-/media/Files/downloads/gnu/13.2.rel1/binrel/arm-gnu-toolchain-13.2.rel1-x86_64-arm-none-eabi.tar.xz
tar -xvf arm-gnu-toolchain-13.2.rel1-x86_64-arm-none-eabi.tar.xz
export PATH="$HOME/arm-gnu-toolchain-13.2.Rel1-x86_64-arm-none-eabi/bin/:$PATH"
git clone --recursive https://github.com/HimaxWiseEyePlus/Seeed_Grove_Vision_AI_Module_V2.git
cd Seeed_Grove_Vision_AI_Module_V2
cd EPII_CM55M_APP_S
gmake clean
gmake
cd ../we2_image_gen_local/
cp ../EPII_CM55M_APP_S/obj_epii_evb_icv30_bdv10/gnu_epii_evb_WLCSP65/EPII_CM55M_gnu_epii_evb_WLCSP65_s.elf input_case1_secboot/
./we2_local_image_gen_macOS_arm64 project_case1_blp_wlcsp.json
```
flash
```
pip install -r xmodem/requirements.txt
python xmodem/xmodem_send.py --port=/dev/tty.usbmodem58C60539941 --baudrate=921600 --protocol=xmodem --file=we2_image_gen_local/output_case1_sec_wlcsp/output.img
picocom /dev/tty.usbmodem58C60539941 -b 921600
```

### gap8

