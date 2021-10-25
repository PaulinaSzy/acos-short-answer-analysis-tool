import dash

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Acos short answer analysis"
server = app.server
