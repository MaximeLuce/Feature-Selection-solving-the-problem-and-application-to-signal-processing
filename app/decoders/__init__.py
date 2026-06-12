from app.decoders.am import AMDecoder
from app.decoders.sigma import SigmaDecoder


def build_decoder(config: dict):
    if not config:
        raise ValueError("Missing decoder config.")

    name = config.get("name")
    if name == "sigma":
        return SigmaDecoder(
            bounds=config.get("bounds"),
            seed=config.get("seed"),
        )
    if name == "am":
        return AMDecoder(bounds=config.get("bounds"))

    valid_names = "sigma, am"
    raise ValueError(f"Unknown decoder: {name}. Use one of: {valid_names}")


__all__ = ["AMDecoder", "SigmaDecoder", "build_decoder"]
