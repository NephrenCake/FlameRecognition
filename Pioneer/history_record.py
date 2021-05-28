import pandas as pd

def history(time,picture_number,ruohuo,zhengchang,guohuo):
    data_ls =pd.read_csv('history.csv',index_col=0,encoding='gbk',low_memory=False)
    data={'time':time,'picture_number':picture_number,'ruohuo':ruohuo,'zhenchang':zhengchang,'guohuo':guohuo}
    data_pd=pd.DataFrame(data,index=[0])
    history_data=data_ls.append(data_pd,ignore_index=True)
    history_data.to_csv('history.csv')

