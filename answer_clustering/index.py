import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction
from app import app
from apps import semantics_app, syntax_app
from apps import callbacks_semantics, callbacks_syntax


app.layout = html.Div(
    id="main-layout",
    children=[dcc.Location(id="url", refresh=False), html.Div(id="page-content")],
)

app.clientside_callback(
    ClientsideFunction("clientside", "addCollapsible"),
    Output("page-content", "data-loaded"),  # Just put some dummy output here
    [
        Input("graphs-loaded-flag", "id"),
    ],  # This will trigger the callback when the object is injected in the DOM
)


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def display_page(pathname):
    if pathname == "/":
        return dcc.Location(pathname="/semantic-error-analysis", id="url")
        # return semantics_app.content_semantic_error
    if pathname == "/semantic-error-analysis":
        return semantics_app.content_semantic_error
    if pathname == "/semantic-progress-analysis":
        return semantics_app.content_semantic_progress
    if pathname == "/syntax-error-analysis":
        return syntax_app.content_syntax_error
    elif pathname == "/syntax-progress-analysis":
        return syntax_app.content_syntax_progress
    else:
        return "404"


if __name__ == "__main__":
    app.run_server(debug=True)
