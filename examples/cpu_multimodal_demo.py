"""CPU-only aligned image/audio/text late-fusion training and model roundtrip."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from everyo.multimodal import (
    TinyMultimodalNet,
    evaluate_multimodal,
    multimodal_loss,
    prepare_multimodal_batch,
)
from everyo.optim import Adam
from everyo.serialization import load, save


def main() -> None:
    images = np.zeros((4, 4, 4, 3), dtype=np.float32)
    images[:2] = 1
    audio = np.zeros((4, 64), dtype=np.float32)
    audio[:2, ::2] = 0.8
    batch = prepare_multimodal_batch(
        images, audio, ["bright", "bright", "dark", "dark"], np.array([1, 1, 0, 0])
    )
    model = TinyMultimodalNet(classes=2, embed_dim=8, seed=12)
    optimizer = Adam(model.parameters(), lr=0.025)
    before = evaluate_multimodal(model, batch)
    for _ in range(35):
        optimizer.zero_grad()
        loss = multimodal_loss(model(batch), batch.labels)
        loss.backward()
        optimizer.step()
    after = evaluate_multimodal(model, batch)
    print(f"CPU multimodal loss: {before['loss']:.4f} -> {after['loss']:.4f}")
    print(f"Training-set accuracy: {after['accuracy']:.0%}")
    with TemporaryDirectory(prefix="everyo-mm-") as directory:
        path = Path(directory) / "multimodal.evo"
        save(model, path)
        restored = load(path)
    np.testing.assert_allclose(model(batch).data, restored(batch).data, atol=1e-6)
    print("Checkpoint round trip: PASS")


if __name__ == "__main__":
    main()
