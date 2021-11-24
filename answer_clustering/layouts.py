from dash import dcc
from dash import html

check_active = lambda act, curr: " active" if (act == curr) else ""

menu = lambda active: html.Div(
    className="sidebar",
    children=[
        html.Div(
            className="sidebar__content",
            children=[
                html.H1(
                    "Clustering of student answers for Acos shortanswer exercises",
                ),
                html.H2(
                    "Syntactic based analysis of order type questions",
                ),
                html.H3(
                    dcc.Link(
                        "Error analysis",
                        href="/syntax-error-analysis",
                        className="nav-link" + check_active(active, "syntax_error"),
                    )
                ),
                html.H3(
                    dcc.Link(
                        "Student's progress analysis",
                        href="/syntax-progress-analysis",
                        className="nav-link" + check_active(active, "syntax_progress"),
                    )
                ),
                html.H2(
                    "Semantic based analysis of style type questions",
                ),
                html.H3(
                    dcc.Link(
                        "Error analysis",
                        href="/semantic-error-analysis",
                        className="nav-link" + check_active(active, "semantic_error"),
                    )
                ),
                html.H3(
                    dcc.Link(
                        "Student's progress analysis",
                        href="/semantic-progress-analysis",
                        className="nav-link"
                        + check_active(active, "semantic_progress"),
                    )
                ),
            ],
        )
    ],
)
