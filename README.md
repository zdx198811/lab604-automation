# lab604-automation

An instrument configuration/automation platform, designed for Nokia Shanghai Bell D604 lab environment.

<<<<<<< HEAD
There are various kinds of 'programmable' devices in our lab, e.g. Keysight oscilloscopes & AWGs, VadaTech chassis, Xilinx FPGA boards, etc. Each device/instrument has its own control interface, it is very time-consuming to manually turn the knobs and push the buttons or to craft separate scripts to make a testbed configuration for one experiment or demo case. A unified software framework that controls all the devices in a centralized and programmable manner will be very necessary.

There are similar requirements for test automation in many R&D labs, and some mature test automation platform solutions must exist already. But our lab focuses more on research & innovation and could be far less standardized than those R&D labs. As our work goes on, experiment (and sometimes demo) requirements change rapidly, so the devices we use and their functionalities/APIs also change constantly. Therefore, we do not need a sophisticated and stable software suit, neither do we care about performance and robustness. The only thing we need is a simple and flexible framework (or template) to glue different devices' control scripts, which can be quickly hacked and re-structured when needed to.

To those who accidentally enter here: I don't think anyone else outside our lab will benefit from these codes directly. Even if you have exactly the same equipment as ours, there are many critical hardware functions may differ and make the scripts inapplicable. For example, the FPGA images and embedded Linux hardware drivers will be different case by case. But still, we make this repository public and open, not only for harvesting the convenience of GitHub utilities but also to inspire anyone trying to develop similar systems (by providing a BAD example, though).
=======
There are various kinds of 'programmable' devices in our lab, e.g. Keysight oscilloscopes/AWGs, VadaTech chassises, Xilinx FPGA boards, etc. Each device/instrument has its own control interface, it is very time-consuming to mannulay turn the knobs and push the buttoms or to craft separate scripts to make a testbed configuration for one experiment or demo. A unified software framework that controls all the divices in a centralized and programmable manner will be very necessary.

There are similar requirements for test automation in many R&D labs, and some mature test automation platform solutions must exist already. But our lab focuses more on research & innovations and could be far less standardized than those R&D labs. As our work going on, experiment (and sometimes demo) requirements change rapidly, so the devices we use and their functionalities/APIs also change constatly. Therefore, we do not need a sophiscated and stable software suit, neither do we care about performance and robustness. The only thing we need is a simple and flexible framework (or template) to glue different devices control scripts, which can be quickly hacked and re-structured when needed to.

To those who accidentally roamed here: 
I don't think anyone else outside our lab will benefit from these codes directly. Even if you have exactly the same equipments as ours, there are many critical hardware functions may differ. For example, the FPGA images and embedded Linux hardware drivers will be different case by case. But still, we make this repository public and open, not only for harvesting the convenience of GitHub utilities, but also to inspire anyone trying to develop similar systems (by providing a BAD example, though).
>>>>>>> e442017cf0c565cb3cc9ff8a03b2970da6f4ea8a

Contact: Dongxu Zhang (dongxu.c.zhang@nokia-sbell.com)

## 1. overall sytem architecture

![system architecture](https://github.com/zdx198811/lab604-automation/blob/dev/doc/images/system_architecture2.png "system architecture")

The system can be roughly divided into the frontend (controller) and backend (lab instruments/devices) parts, interconnected via Ethernet. The frontend consists of one or several controllers, on which applications can be flexibly built upon unified device APIs. The backend may be implemented with very different architectures, for example, Keysight devices typically support standard VISA interface, while VadaTech devices are implemented to be controllable via a very simple socket interface. It is the corresponding frontend modules that wrap the different APIs into abstract devices classes with unified APIs.

More specifically, at the frontend, the user application may instantiate different device objects and call their `getData(*args)` or `config(*args)` methods without caring about the underlying communication details of different devices. Devices with VISA interface (Keysight AWG and oscilloscope) are abstracted as `visa_Devices`, Vadatech devices are typically abstracted as `vt_devices`, and Xilinx devices are abstracted as `Xilinx_devices`. Other devices can be added by creating corresponding classes in the frontend.

### 1.1 Example: the fronthaul/PON demo GUI

The following figure shows the key moduels used in the fronthaul/PON demo.
![code modules](https://github.com/zdx198811/lab604-automation/blob/dev/doc/images/code_modules.png "code modules")

### 1.2 usage instruction

On the vadatech device, simply run: `python vt_device_backend.py dev [-s]`.
where `dev` is the name of the device, e.g. vt855 or vt899fh (note there should be corresponding modules in the _labdevices_ folder), and `-s` specifies whether to use locally saved data for simulation.

For the frontend, usually a GUI application, see examples like _GUIMainTest-vt855.py_ and _GUIMainTest-vt899fh.py_.

