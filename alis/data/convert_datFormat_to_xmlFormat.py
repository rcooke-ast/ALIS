astr = "  <TR>\n    <TD>32</TD>\n    <TD>32.066</TD>\n    <TD>0.0</TD>\n    <TD>S</TD>\n    <TD>I</TD>\n    <TD>WWW</TD>\n    <TD>FFF</TD>\n    <TD>GGG</TD>\n    <TD>NaN</TD>\n    <TD>NaN</TD>\n  </TR>"

lines = ["S I   1474.3785 0.0148 4.54e7",
"S I   1425.2191 0.00148 8.12e6",
"S I   1425.1879 0.0223 7.31e7",
"S I   1425.0300 0.125 2.92e8",
"S I   1401.5142 0.0128 7.22e7",
"S I   1316.6219 0.000358 2.30e6",
"S I   1316.6150 0.00530 2.04e7",
"S I   1316.5425 0.0279 7.67e7",
"S I   1303.4300 0.00436 2.85e7",
"S I   1296.1739 0.022 1.46e8"]

for ll in range(len(lines)):
    linspl = lines[ll].split()
    print(astr.replace("WWW",linspl[2]).replace("FFF",linspl[3]).replace("GGG",linspl[4]))
print("Number of lines added:", len(lines))
