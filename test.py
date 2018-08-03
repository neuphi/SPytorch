import visdom
vis = visdom.Visdom()

trace = dict(x=[1, 2, 3], y=[4, 5, 6], mode="markers+lines", type='custom', marker={'color': 'red', 'symbol': 104, 'size': "10"}, text=["one", "two", "three"], name='1st Trace')

layout = dict(title="Loss Function", xaxis={'title': 'Epochs'}, yaxis={'title': 'Value'})

vis._send({'data': [trace], 'layout': layout, 'win': 'mywin'})
