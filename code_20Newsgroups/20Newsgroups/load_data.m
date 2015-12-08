function data = load_data(setting)


data = eval(sprintf('load_%s', setting));
data.setting = setting;