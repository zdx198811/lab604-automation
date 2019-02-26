# lab604-automation

## overall sytem architecture

![system architecture](https://github.com/zdx198811/lab604-automation/blob/dev/doc/images/system_architecture2.png "system architecture")

There are various kinds of devices in the lab, e.g. VadaTech chassises, Xilinx FPGA boards, and Keysight instruments. Each has its own control interface. This project is to provide a unified remote programming platform, faciliating centralized experiment configuration, and convenient demo development.

The system can be roughly divided into the frontend and backend parts, interconnected via Ethernet. The frontend consisits of a controller, on which applications can be flexibly built upon unified device APIs. The backend may be implemented with very different architecture, for example, Keysight devices typically support standard VISA interface, while VadaTech devices are implemented to be controllable via a very simple socket interface. It is the corresponding frontend modules that wraps the different APIs into abstract devices classes with relatively unified APIs.

More specifically, at the frontend, the user application may instantiate different divice objects and call their `getData(*args)` or `config(*args)` methods without caring about the underlying communication details of different device. Devices with VISA interface (Keysight AWG and oscilloscope) are abstracted as `visa_Devices`, Vadatech devices are typically abstracted as `vt_devices`, and Xilinx devices are abstracted as `Xilinx_devices`. Other devices can be added by creating corresponding classes in the frontend.

### Example: the fronthaul/PON demo GUI

The following figure shows the key moduels used in the fronthaul/PON demo.
![code modules](https://github.com/zdx198811/lab604-automation/blob/dev/doc/images/code_modules.png "code modules")

### example usage instruction

On the vadatech device, simply run: `python vt_device_backend.py dev [-s]`.
where `dev` is the name of the device, e.g. vt855 or vt899fh (note there should be corresponding modules in the _labdevices_ folder), and `-s` specifys whether to use locally saved data for simulation.

For the frontend, usually a GUI application, see examples like _GUIMainTest-vt855.py_ and _GUIMainTest-vt899fh.py_.

