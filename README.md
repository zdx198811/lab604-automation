# lab604-automation

## overall sytem architecture

![system architecture pic](https://github.com/zdx198811/lab604-automation/tree/dev/doc/images/system_architecture.png "system architecture")

## key modules of the code

![code modules](https://github.com/zdx198811/lab604-automation/tree/dev/doc/images/code_modules.png "code modules")

## usage instruction

For vadatech device backend, simply run: `python vt_device_backend.py dev [-s]`.
where 
`dev` is the name of the device, e.g. vt855 or vt899fh (note there should be corresponding modules in the _labdevices_ folder), and `-s` specifys whether to use locally saved data for simulation.

For the frontend, usually a GUI application, see examples like _GUIMainTest-vt855.py_ and _GUIMainTest-vt899fh.py_.

