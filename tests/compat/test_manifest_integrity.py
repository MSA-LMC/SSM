# Detect accidental edits to the migrated expression split manifests.
import hashlib
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXPRESSION_SPLITS = REPOSITORY_ROOT / "splits" / "emotion"

EXPECTED = {
    "DFEW/DFEW_set_1_test.txt": (
        2341,
        "970fa8056f89f5444fe0f41da833593e92f46de119387dcbe8b2c1d184f37c48",
    ),
    "DFEW/DFEW_set_1_train.txt": (
        9356,
        "fc424f446942970702b6ef30a54b7344b26388061d0f4c8031bdf24082d4f073",
    ),
    "DFEW/DFEW_set_2_test.txt": (
        2341,
        "3348da4bc178488504421d0c5cb843c84337b425737b2d5857f09b9c4d379124",
    ),
    "DFEW/DFEW_set_2_train.txt": (
        9356,
        "125f823a4bee1d4995de4c6bde28a2e1c9c65fb3ae37bab588f9d60ac77f7227",
    ),
    "DFEW/DFEW_set_3_test.txt": (
        2340,
        "f6c3a4621d6162270d9a45593eea9655acaea07c5adfcd0e3ef5c6eb85bf56b1",
    ),
    "DFEW/DFEW_set_3_train.txt": (
        9357,
        "128570cacf98a4ef48bf59046fa757290094ff681da0cc9eaa6107ad2bb1e3c3",
    ),
    "DFEW/DFEW_set_4_test.txt": (
        2339,
        "ba0d5926ae8f9cccc7c72571542eb345ff9ef1506a05e3ca45bfe01eac00fdd7",
    ),
    "DFEW/DFEW_set_4_train.txt": (
        9358,
        "859f10bb55baa0e6354926de2b1c01627614d146d47f18c719771e1d6dd8345a",
    ),
    "DFEW/DFEW_set_5_test.txt": (
        2336,
        "5f6d8e898e5aeedfa3c1f0749085cbf566ccaa0696c0146cf7e2ed3e652a1eae",
    ),
    "DFEW/DFEW_set_5_train.txt": (
        9361,
        "b122fbb6fba74b79836c1b1c490f3023d68be7f9b760930893d6efc0cc2a0c88",
    ),
    "FERV39K/FERV39K_test.txt": (
        7847,
        "f02351b9dd0c399eb248b45384e89e4f469d3404d19e0816c6b18854599aee2e",
    ),
    "FERV39K/FERV39K_train.txt": (
        31088,
        "79acdca3ec57db27a7d7e0d36ecadd62e5bc9d64925d62e1d93361bc1d37da9d",
    ),
    "MAFW/MAFW_set_1_test.txt": (
        1839,
        "d1749f5d065d9e6131f901382bb428aca3f9bf44610570bf44e3759cb00eceff",
    ),
    "MAFW/MAFW_set_1_train.txt": (
        7333,
        "d51532f1d86049f0b46ebed9d739f4e4183998a550c8d6eca6a80e5752e40f37",
    ),
    "MAFW/MAFW_set_2_test.txt": (
        1837,
        "b2fa92452f9aded2e5ec1305b42cc8470ad0cac41384c911a48259d2602d421a",
    ),
    "MAFW/MAFW_set_2_train.txt": (
        7335,
        "2b3aac84cd5769f9140f9f109c160804f96dc6e3c752bc3501ee55f67042e215",
    ),
    "MAFW/MAFW_set_3_test.txt": (
        1833,
        "e9e58dfb731bcef92a05557995754e7272439274b3708509e44bbec93f90100f",
    ),
    "MAFW/MAFW_set_3_train.txt": (
        7339,
        "38a68529796b64b17c08a9b27070bafb28db78bf5dd661d1d49c5932def43851",
    ),
    "MAFW/MAFW_set_4_test.txt": (
        1832,
        "16d2b8f6cfcf29825cfa366dfc9bac58430ee2e0ade423bc578d0af85a83f7ce",
    ),
    "MAFW/MAFW_set_4_train.txt": (
        7340,
        "5409df9f400f0c64446f25d98563753a509e85c3344319dcb677366ab831eba8",
    ),
    "MAFW/MAFW_set_5_test.txt": (
        1831,
        "1cd3c91fbefc35f620eed73c11716a3055ebef709c189db313618973699757f5",
    ),
    "MAFW/MAFW_set_5_train.txt": (
        7341,
        "7f1163649f98acdd21b61d2ebd6b415329b4e41b35468d8fa920954209e71026",
    ),
}


def test_expression_manifests_match_the_migrated_records():
    for relative_path, (expected_lines, expected_digest) in EXPECTED.items():
        path = EXPRESSION_SPLITS / relative_path
        lines = path.read_text(encoding="utf-8").splitlines()
        normalized = ("\n".join(lines) + "\n").encode()
        assert len(lines) == expected_lines
        assert hashlib.sha256(normalized).hexdigest() == expected_digest
