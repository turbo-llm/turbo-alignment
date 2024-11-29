import typer

app = typer.Typer(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
    pretty_exceptions_enable=False,
)
