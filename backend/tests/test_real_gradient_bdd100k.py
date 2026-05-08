import json

import pytest

from policy_agent.analysis.real_gradient_benchmark import (
    RealGradientBenchmarkConfig,
    load_bdd100k_clients,
    load_real_clients,
    run_real_gradient_benchmark,
)


def _write_image(path, color):
    image_module = pytest.importorskip("PIL.Image")
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image_module.new("RGB", (12, 8), color=color)
    image.save(path)


def _make_bdd_fixture(root):
    image_root = root / "images" / "100k" / "train"
    records = []
    specs = [
        ("clear_day_1.jpg", "clear", "daytime", "city street", (220, 220, 220)),
        ("clear_day_2.jpg", "clear", "daytime", "city street", (200, 200, 200)),
        ("rain_night_1.jpg", "rainy", "night", "highway", (30, 30, 80)),
        ("rain_night_2.jpg", "rainy", "night", "highway", (35, 35, 90)),
        ("snow_day_1.jpg", "snowy", "daytime", "residential", (240, 240, 255)),
        ("snow_day_2.jpg", "snowy", "daytime", "residential", (245, 245, 255)),
    ]
    for name, weather, timeofday, scene, color in specs:
        _write_image(image_root / name, color)
        records.append(
            {
                "name": name,
                "attributes": {
                    "weather": weather,
                    "timeofday": timeofday,
                    "scene": scene,
                },
                "labels": [],
            }
        )

    label_file = root / "labels" / "bdd100k_labels_images_train.json"
    label_file.parent.mkdir(parents=True, exist_ok=True)
    label_file.write_text(json.dumps(records), encoding="utf-8")
    return label_file, image_root


def test_load_bdd100k_clients_groups_attribute_pseudo_clients(tmp_path):
    label_file, image_root = _make_bdd_fixture(tmp_path)

    clients, info = load_bdd100k_clients(
        tmp_path,
        label_file=str(label_file),
        image_dir=str(image_root),
        image_size=8,
        target_attribute="weather",
        client_group="weather_timeofday",
        corner_values="rainy,snowy",
        max_clients=10,
        min_samples_per_client=2,
        max_samples_per_client=2,
        seed=7,
    )

    assert info["source"] == "bdd100k"
    assert info["target_attribute"] == "weather"
    assert info["real_client_partitions"] is False
    assert set(info["class_to_id"]) == {"clear", "rainy", "snowy"}
    assert {
        info["id_to_class"][label]
        for label in info["corner_labels"]
    } == {"rainy", "snowy"}
    assert len(clients) == 3
    assert all(client.inputs.shape == (2, 8 * 8 * 3) for client in clients)
    assert all(client.targets.shape == (2,) for client in clients)
    assert all(client.metadata["client_group"] == "weather_timeofday" for client in clients)


def test_load_real_clients_supports_bdd100k_source(tmp_path):
    label_file, image_root = _make_bdd_fixture(tmp_path)
    config = RealGradientBenchmarkConfig(
        source="bdd100k",
        bdd_data_dir=str(tmp_path),
        bdd_label_file=str(label_file),
        bdd_image_dir=str(image_root),
        bdd_image_size=8,
        bdd_corner_values="rainy,snowy",
        max_clients=2,
        min_samples_per_client=2,
        max_samples_per_client=2,
    )

    clients, info = load_real_clients(config)

    assert len(clients) == 2
    assert info["source"] == "bdd100k"
    assert info["corner_labels"]


def test_bdd100k_source_runs_real_gradient_smoke(tmp_path):
    label_file, image_root = _make_bdd_fixture(tmp_path)
    config = RealGradientBenchmarkConfig(
        source="bdd100k",
        bdd_data_dir=str(tmp_path),
        bdd_label_file=str(label_file),
        bdd_image_dir=str(image_root),
        bdd_image_size=8,
        bdd_corner_values="rainy,snowy",
        max_clients=3,
        min_samples_per_client=2,
        max_samples_per_client=2,
        clients_per_round=3,
        rounds=1,
        pretrain_steps=1,
        local_batch_size=2,
        attack_fraction=0.0,
        corner_harm_fraction=0.0,
        noise_fraction=0.0,
    )

    result = run_real_gradient_benchmark(config)

    assert result["dataset"]["source"] == "bdd100k"
    assert result["dataset"]["input_dim"] == 8 * 8 * 3
    assert result["methods"]["cornerdrive"]["round_records"]
