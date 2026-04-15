import warnings

# Silence only the specific FutureWarning about weights_only=False
warnings.filterwarnings(
    "ignore",
    message=r"You are using `torch\.load` with `weights_only=False`.*",
    category=FutureWarning,
)


from .run_md_opt import main

if __name__ == "__main__":
    main()

