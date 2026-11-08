"""The reference plan of the tests, on a neutral place: 45.0 N 10.0 E (a field, no site).

The planner works in local metres (111 320 m per degree, cos(lat) for the longitude), so a rectangle
given in metres gives the same strips, shots and times wherever it is put.
"""
from ppg_flightplan import geometry

CENTRE = (45.0, 10.0)


def rectangle(width_m: float, height_m: float, centre=CENTRE) -> list:
    """A rectangle of ``width_m`` x ``height_m`` (east x north) around ``centre``, in the planner's metres."""
    d_lat = height_m / 2 / geometry.METRES_PER_DEGREE
    d_lon = width_m / 2 / geometry.metres_per_degree_lon(centre[0])
    return [(centre[0] - d_lat, centre[1] - d_lon), (centre[0] - d_lat, centre[1] + d_lon),
            (centre[0] + d_lat, centre[1] + d_lon), (centre[0] + d_lat, centre[1] - d_lon)]


def block_a(mode="intervallo", high=True) -> dict:
    """The nadir block of the reference plan: 846 x 822 m at 110 m, 80/70 % overlaps, 5 m/s."""
    return {"tipo": "griglia", "nome": "A nadirale", "area": rectangle(846, 822), "quota": 110,
            "inclinazione": -90, "avanti": 80, "lato": 70, "orientamento": 0, "velocita": 5, "modo": mode,
            "alta": high}


def reference_plan() -> dict:
    """The nadir block and a ring of 200 m at 75 m looking at the centre."""
    return {"nome": "Campo di prova", "camera": "mini4pro", "alta": True, "minutiBatteria": 18,
            "blocchi": [block_a(), {"tipo": "anello", "nome": "D anello alto", "centro": CENTRE, "raggio": 200,
                                    "quota": 75, "inclinazione": -35, "avanti": 80, "velocita": 4, "alta": False}]}
