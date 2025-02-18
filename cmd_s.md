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
cd npu/milk-v
cd sample
./compile_samples.sh
scp -O sample_{model} root@192.168.42.1:/root/
scp -O ../../model/{model}/{model}.cvimage
ssh root@192.168.42.1
cd root
./sample_{model} {model}.cvimodel
```

### canmv
build boot image (linux)
```
docker pull ghcr.io/kendryte/k230_sdk
docker images | grep ghcr.io/kendryte/k230_sdk
cd k230_sdk
git clone -b v1.0.1 --single-branch https://github.com/kendryte/k230_sdk.git
cd k230_sdk
make prepare_sourcecode
docker run -u root -it -v $(pwd):$(pwd) -v $(pwd)/toolchain:/opt/toolchain -w $(pwd) ghcr.io/kendryte/k230_sdk /bin/bash
make CONF=k230_canmv_defconfig
```
flash boot image with ```sudo dd if=sysimage-sdcard.img of=/dev/sdc bs=1m oflag=sync```

build app image (add app path to build_app_sub.sh) (linux)
```
sudo apt-get update
sudo apt-get install -y dotnet-sdk-7.0
cd k230_sdk
docker run -u root -it -v $(pwd):$(pwd) -v $(pwd)/toolchain:/opt/toolchain -w $(pwd) ghcr.io/kendryte/k230_sdk /bin/bash
cd src/reference/ai_poc
./build_app.sh {app_name}
```
build written to ```k230_sdk/src/reference/ai_poc/k230_sdk```

flash app image
```
picocom /dev/ttyACM1
scp -r username@domain_or_IP:source_directory destination_directory_on_board
```