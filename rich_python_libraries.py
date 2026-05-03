import typer
from rich.console import Console
console = Console()
console.print("[bold magenta]Hello, world![/bold magenta] ✨")






# typer 

app = typer.Typer()

@app.command()
def greet(name: str):
    print(f"Hello {name}!")
if __name__ == "__main__":
    app()