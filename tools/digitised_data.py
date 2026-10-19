"""Digitised experimental data, kept as raw pixel readings plus axis calibrations.

Run this file to regenerate gasmem/data/*.csv.  Keeping the pixels makes the
digitisation auditable: anyone with the PDF can check a point.

How the points were read
------------------------
Pan (1986) is a scanned article: pages were rendered at 400 dpi with pdftoppm and
cropped.  Feng et al. (1999) contain embedded 300-ppi figure images, extracted
with pdfimages.  Axis frames and tick marks were located from the pixel data
(rows/columns of dark pixels), so the calibration uses detected tick positions,
not visual guesses.  Marker centres were found by morphological filtering
(filled markers) and hole filling (open markers); markers touching a curve or
another marker were located by eye on magnified, pixel-gridded crops.
Typical reading uncertainty is about +/-3 px, i.e. about +/-0.1 % in
composition and +/-0.002 in stage cut.

The markers show measurements; the curves in the figures (models or fitted
lines) are not digitised.
"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "gasmem" / "data"


def lin(p, p0, v0, p1, v1):
    """Linear pixel -> value map through two calibration points."""
    return v0 + (p - p0) * (v1 - v0) / (p1 - p0)


rows = []


def add(source, figure, quantity, series, x_name, pts, fx, fy, **meta):
    for px, py in pts:
        rows.append(dict(source=source, figure=figure, quantity=quantity, series=series,
                         x_name=x_name, x=round(fx(px), 5), y=round(fy(py), 5),
                         px=px, py=py, **meta))


# ---------------------------------------------------------------------------
# Pan (1986), Fig. 2: H2 recovery from NH3 purge gas, 6964 kPa / 1123 kPa, 25 C.
# Crop of page 4 at 400 dpi.  x: ticks 0.4/0.5/0.6 at 564.5/846.5/1128.5 px.
# Left axis (permeate H2 %): 90 at 346.5 px, 27.95 px per %.
# Right axis (H2 recovery %): 100 at 66.5 px, 40 at 738.5 px.
# filled = countercurrent run, open = cocurrent run.
th2 = lambda x: lin(x, 564.5, 0.4, 846.5, 0.5)
pur2 = lambda y: 90 - (y - 346.5) / 27.95
rec2 = lambda y: lin(y, 66.5, 100, 738.5, 40)
P = "Pan1986"
add(P, "Fig. 2", "permeate H2 %", "countercurrent", "stage cut",
    [(397.5, 148.75), (505, 158.75), (598.75, 173.75), (793.3, 211.7), (944, 317), (1010.5, 388)],
    th2, pur2)
add(P, "Fig. 2", "permeate H2 %", "cocurrent", "stage cut",
    [(442.5, 158.75), (747.5, 211.7), (760.8, 216.7), (925, 314.5), (997, 440), (1126.5, 608)],
    th2, pur2)
add(P, "Fig. 2", "H2 recovery %", "countercurrent", "stage cut",
    [(396.5, 444.5), (503.5, 366.5), (597.5, 300), (792.5, 165), (943, 97), (1009.5, 81)],
    th2, rec2)
add(P, "Fig. 2", "H2 recovery %", "cocurrent", "stage cut",
    [(445.5, 409.5), (747.5, 202.5), (762, 191), (928, 108), (997.5, 117), (1126.5, 111.5)],
    th2, rec2)

# Pan (1986), Fig. 3: impurities in the permeate, same runs as Fig. 2.
# x: ticks 0.5/0.6 at 903.5/1225.5 px.  y: 0 % at 797 px, 63.6 px per %.
th3 = lambda x: lin(x, 903.5, 0.5, 1225.5, 0.6)
c3 = lambda y: (797 - y) / 63.6
xs_cc = [391, 514, 619, 840, 981, 1095]          # countercurrent runs
xs_co = [446, 789, 805, 995, 1076, 1223]         # cocurrent runs
fig3 = {
    "N2": {"countercurrent": [708.75, 697.5, 680, 639, 519.5, 438],
           "cocurrent": [701, 639, 626, 519.5, 380.5, 185.5]},
    "CH4": {"countercurrent": [731, 724.5, 714, 680, 599, 539],
            "cocurrent": [726, 684, 677, 596, 495, 349]},
    "Ar": {"countercurrent": [764.5, 761, 754, 745, 705, 682.5],
           "cocurrent": [763, 746, 745, 707.5, 666, 617]},
}
for gas, d in fig3.items():
    for series, ys in d.items():
        xs = xs_cc if series == "countercurrent" else xs_co
        add(P, "Fig. 3", f"permeate {gas} %", series, "stage cut", list(zip(xs, ys)), th3, c3)

# Pan (1986), Fig. 6: CO2 removal from C1-C3 hydrocarbons, 3528 kPa / 92.8 kPa,
# 10 C, countercurrent.  Stacked axes sharing ticks at 218 / 467.5 / 715.5 px:
# CO2 90/80/70 %, CH4 30/20/10 %, C2H6 and C3H8 12/8/4 %; CH4, C2H6, C3H8 are
# zero at the bottom frame (965 px).  x: 0.4 at 400.5 px, 252.8 px per 0.1.
th6 = lambda x: lin(x, 400.5, 0.4, 652.5, 0.5)
fig6 = {
    "CO2": ([(453, 88), (528, 103), (570, 106), (673, 152), (793, 288), (920, 454)],
            lambda y: lin(y, 218, 90, 715.5, 70)),
    "CH4": ([(455, 862), (530, 853), (572, 848), (673, 807), (793, 693), (920, 557)],
            lambda y: lin(y, 965, 0, 218, 30)),
    "C2H6": ([(455, 915), (530, 914), (572, 914), (793, 838), (920, 773)],
             lambda y: lin(y, 965, 0, 218, 12)),
    "C3H8": ([(455, 958), (530, 958), (572, 956), (669, 950), (793, 944), (920, 936)],
             lambda y: lin(y, 965, 0, 218, 12)),
}
for gas, (pts, fy) in fig6.items():
    add(P, "Fig. 6", f"permeate {gas} %", "countercurrent", "stage cut", pts, th6, fy)

# Pan (1986), Fig. 7: H2S/CO2 from sour gas, 5245 kPa / 92.8 kPa, 50 C,
# countercurrent.  x: 0.2 at 186.5 px, 0.8 at 1191 px.
# Left axis (H2S, CH4): 20 % at 1107.5 px, 16.75 px per %.
# Right axis (CO2, N2): 0 % at 1107.5 px, 83.75 px per %.
th7 = lambda x: lin(x, 186.5, 0.2, 1191, 0.8)
left7 = lambda y: 20 + (1107.5 - y) / 16.75
right7 = lambda y: (1107.5 - y) / 83.75
add(P, "Fig. 7", "permeate H2S %", "countercurrent", "stage cut",
    [(409, 433), (455, 480), (500, 522), (545, 588), (607, 625), (623, 633), (722, 727), (803, 790)],
    th7, left7)
add(P, "Fig. 7", "permeate CH4 %", "countercurrent", "stage cut",
    [(405, 982), (450, 942), (500, 896), (545, 851), (613, 778), (623, 780), (720, 688), (806, 580)],
    th7, left7)
add(P, "Fig. 7", "permeate CO2 %", "countercurrent", "stage cut",
    [(406, 322), (450, 350), (498, 378), (545, 428), (607, 476), (618, 499.6), (722, 553), (808, 616)],
    th7, right7)
add(P, "Fig. 7", "permeate N2 %", "countercurrent", "stage cut",
    [(405, 797), (450, 775), (500, 748), (545, 720), (611, 683), (622, 667), (722, 620), (806, 580)],
    th7, right7)

# ---------------------------------------------------------------------------
# Feng et al. (1999), Figs. 4 and 5: air, 690 kPa(g) feed, 0 psig permeate
# (local atmosphere 70 cmHg), 23 C.  Embedded 300-ppi images.
F = "Feng1999"
th4 = lambda x: lin(x, 101, 0.2, 702, 0.9)
n2_4 = lambda y: lin(y, 11, 100, 861, 85)
fig4 = {
    "bore-side feed, countercurrent": [(657.3, 44.9), (601.2, 47.5), (534.8, 79.4), (455.7, 156.8),
                                       (442.8, 168.3), (376.5, 269), (338.3, 329.5), (321.8, 357.3),
                                       (266.1, 457.9), (235.7, 520.8), (199.7, 583.6), (175.6, 624.6)],
    "shell-side feed, countercurrent": [(629.8, 39), (509.4, 96.7), (428.6, 195.2), (367.6, 289.1),
                                        (312.4, 389.2), (260.7, 496.7), (220.5, 563.5), (182.5, 642.2)],
    "bore-side feed, cocurrent": [(656.9, 208), (589.7, 243.1), (507.1, 288.4), (428.6, 338.5),
                                  (383, 388), (330.4, 441.7), (303.8, 480.6), (256.8, 544.8),
                                  (232.9, 580.6), (186.1, 656.1)],
    "shell-side feed, cocurrent": [(630, 244), (528, 289), (448.4, 346), (415.3, 373.6), (374.6, 411.7),
                                   (258.7, 557), (197.1, 653.7), (171.7, 704.8), (145.6, 751.1)],
}
for series, pts in fig4.items():
    add(F, "Fig. 4", "residue N2 %", series, "stage cut", pts, th4, n2_4)

th5 = lambda x: lin(x, 84, 0.2, 684, 0.9)
o2_5 = lambda y: lin(y, 12, 45, 861, 20)
fig5 = {
    "bore-side feed, countercurrent": [(157, 54.4), (180, 83), (216.8, 127.8), (246.1, 166.8),
                                       (302.8, 245.8), (319, 283), (357.9, 344.9), (426.5, 451.6),
                                       (439.5, 465.5), (517.3, 578.6), (584, 658), (638.8, 718.3)],
    "shell-side feed, countercurrent": [(164.7, 90.8), (201.8, 135), (242.5, 182.6), (294.6, 256.3),
                                        (350.3, 336.7), (411.4, 423.5), (493.6, 539.8), (611.3, 687.8)],
    "bore-side feed, cocurrent": [(166.4, 128.6), (217.8, 196.2), (241.4, 232.5), (288.3, 306.6),
                                  (314, 352), (367.4, 421.1), (411.2, 493.2), (489.4, 575.8),
                                  (569.8, 678.6), (639.3, 737.8)],
    "shell-side feed, cocurrent": [(128.8, 87), (154.5, 126.4), (180.1, 172.2), (240.2, 260.6),
                                   (357.2, 425.1), (396.9, 481), (430.7, 519.5), (509.6, 619.4),
                                   (611.5, 719.3)],
}
for series, pts in fig5.items():
    add(F, "Fig. 5", "permeate O2 %", series, "stage cut", pts, th5, o2_5)

# Feng et al. (1999), Fig. 13: N2 production, bore-side feed countercurrent,
# 100 psig, 23 and 45 C.  x: 88 % at 113 px, 100 % at 713 px.
# Top panel recovery: 90 % at 12 px, 10 % at 375 px.
# Bottom panel productivity, Nm3/(m2 h): 0.20 at 398.5 px, 0 at 759.5 px.
n2x = lambda x: lin(x, 113, 88, 713, 100)
rec13 = lambda y: lin(y, 12, 90, 375, 10)
prod13 = lambda y: lin(y, 759.5, 0.0, 398.5, 0.20)
add(F, "Fig. 13", "N2 recovery %", "23 C", "residue N2 %",
    [(170.3, 58.6), (207.5, 68.1), (262, 86), (317, 97.4), (405, 131.2), (430.1, 139.8), (483.5, 158.9),
     (573.6, 198.5), (583.5, 205.1), (653, 256), (679.6, 297.3), (686, 333)], n2x, rec13)
add(F, "Fig. 13", "N2 recovery %", "45 C", "residue N2 %",
    [(289.1, 113.8), (333, 127), (384.2, 144.7), (442, 164), (518, 199.8), (526, 199.8),
     (575.8, 228.9), (636.7, 271.4), (682.6, 330.6)], n2x, rec13)
add(F, "Fig. 13", "N2 productivity Nm3/(m2 h)", "23 C", "residue N2 %",
    [(171.2, 556), (212, 582), (262, 612.1), (317.3, 634.4), (406, 663.8), (431.4, 679.6), (484, 694.5),
     (572.7, 713.5), (584.3, 713.5), (652.9, 730.5), (677.4, 741), (686.3, 748.1)], n2x, prod13)
add(F, "Fig. 13", "N2 productivity Nm3/(m2 h)", "45 C", "residue N2 %",
    [(289.5, 534.2), (334.1, 564.5), (384.9, 594.6), (442.8, 622.2), (519, 659.4), (526, 659.4),
     (575.8, 681.3), (636.5, 706.7), (682.8, 733)], n2x, prod13)

# Feng et al. (1999), Fig. 14: O2 enrichment, same runs.
# x: 20 % at 110.5 px, 45 % at 709 px.  Top: 100 % at 13 px, 50 % at 375 px.
# Bottom: 0.05 at 399.5 px, 0 at 759.5 px.
o2x = lambda x: lin(x, 110.5, 20, 709, 45)
rec14 = lambda y: lin(y, 13, 100, 375, 50)
prod14 = lambda y: lin(y, 759.5, 0.0, 399.5, 0.05)
add(F, "Fig. 14", "O2 recovery %", "23 C", "permeate O2 %",
    [(255, 18), (310, 18), (390.1, 47.3), (397.5, 25.8), (475, 90.2), (516.8, 117.3), (544.5, 127.9),
     (599.6, 183.8), (628.2, 218.4), (661.4, 259.2), (680.7, 287.6)], o2x, rec14)
add(F, "Fig. 14", "O2 recovery %", "45 C", "permeate O2 %",
    [(211.5, 19), (286.3, 26), (344.5, 46.2), (388.1, 66.6), (445.2, 104.2), (480.1, 133.2),
     (511.4, 163), (536.8, 189.6)], o2x, rec14)
add(F, "Fig. 14", "O2 productivity Nm3/(m2 h)", "23 C", "permeate O2 %",
    [(210.8, 692.7), (253.6, 688.1), (309.4, 681.6), (388, 664), (397.7, 667.9), (474.5, 655.9),
     (516.5, 648.2), (544, 631), (599, 620), (628, 612), (661.1, 607.2), (680, 599.9)], o2x, prod14)
add(F, "Fig. 14", "O2 productivity Nm3/(m2 h)", "45 C", "permeate O2 %",
    [(211.2, 621.9), (285.1, 597.2), (344, 578), (387.8, 562.9), (445, 542), (479.9, 527),
     (511, 513), (536.1, 500.9)], o2x, prod14)


# ---------------------------------------------------------------------------
# Makaruk et al. (2010), Figs. 15 and 16: MODEL curves (not measurements) for
# the two-stage cascade with recycle, 98 % CH4 product, as a function of the
# membrane ideal selectivity.  "5 %" and "15 %" are the heat demands, which
# correspond to methane recoveries of 93.8 % and 81.2 %.  Embedded 200-ppi
# images; curves traced column by column (dark pixels, value < 110) at the
# selectivities listed, skipping columns where a curve label overlaps a curve.
# Fig. 15 axes: x 0 at 65.5 px, 90 at 680 px; y 0 at 537.5 px, 0.8 at 5.5 px.
# Fig. 16 axes: x 0 at 76 px, 90 at 678.5 px; y 0 at 537.5 px, 4000 at 5.5 px.
M = "Makaruk2010"
a15 = lambda x: lin(x, 65.5, 0, 680, 90)
w15 = lambda y: lin(y, 537.5, 0, 5.5, 0.8)
a16 = lambda x: lin(x, 76, 0, 678.5, 90)
A16 = lambda y: lin(y, 537.5, 0, 5.5, 4000)
px15 = lambda a: round(65.5 + a * (680 - 65.5) / 90, 1)
px16 = lambda a: round(76 + a * (678.5 - 76) / 90, 1)
py15 = lambda v: round(537.5 - v * (537.5 - 5.5) / 0.8, 1)
py16 = lambda v: round(537.5 - v * (537.5 - 5.5) / 4000, 1)
fig15 = {
    "recovery 93.8 % (5 % heat)": {7: 0.7038, 8: 0.6248, 9: 0.5263, 10: 0.4609, 16: 0.3098, 18: 0.2902,
                                   20: 0.2759, 22: 0.2624, 25: 0.2489, 30: 0.2338, 35: 0.2233,
                                   40: 0.2158, 45: 0.2113, 50: 0.2075, 60: 0.2030, 80: 0.1932},
    "recovery 81.2 % (15 % heat)": {5: 0.5519, 6: 0.5105, 16: 0.2451, 18: 0.2383, 20: 0.2323,
                                    22: 0.2278, 25: 0.2218, 30: 0.2143, 35: 0.2098, 40: 0.2060,
                                    45: 0.2030, 50: 0.2008, 60: 0.1977, 80: 0.1932},
}
for series, d in fig15.items():
    add(M, "Fig. 15", "specific compression power kWh/m3 (model curve)", series, "selectivity",
        [(px15(a), py15(v)) for a, v in d.items()], a15, w15)
fig16 = {
    "recovery 93.8 % (5 % heat)": {7: 3466, 8: 3199, 9: 2936, 10: 2722, 12: 2489, 14: 2323, 16: 2226,
                                   18: 2177, 20: 2147, 22: 2132, 25: 2132, 30: 2154, 35: 2199,
                                   50: 2417, 60: 2586, 80: 2932},
    "recovery 81.2 % (15 % heat)": {5: 2436, 6: 2312, 7: 2135, 8: 1985, 9: 1868, 10: 1767, 12: 1695,
                                    14: 1695, 16: 1703, 18: 1726, 20: 1763, 22: 1786, 25: 1823,
                                    30: 1861, 35: 1812, 50: 1974, 60: 2086, 80: 2320},
}
for series, d in fig16.items():
    add(M, "Fig. 16", "total membrane area m2 (model curve)", series, "selectivity",
        [(px16(a), py16(v)) for a, v in d.items()], a16, A16)


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    for src, d in df.groupby("source"):
        path = OUT / f"{src.lower()}_digitised.csv"
        d.to_csv(path, index=False)
        print(f"wrote {path.relative_to(OUT.parents[1])}: {len(d)} points")


if __name__ == "__main__":
    main()
