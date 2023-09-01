def prompt_user(message) -> bool:
    response = input(f"{message} (y/n): ").strip().lower()
    return response == "y"
