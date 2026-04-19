import pandas as pd
import xml.etree.ElementTree as ET
import os

print('cwd', os.getcwd())
print('files exist', os.path.exists('Utsm.gpx'), os.path.exists('telemetry_20260411_112302.csv'))
tele = pd.read_csv('telemetry_20260411_112302.csv')
print('telemetry columns', tele.columns.tolist())
print('telemetry nulls', tele.isnull().sum().to_dict())
print('telemetry head:')
print(tele.head())

ns = {'gpx': 'http://www.topografix.com/GPX/1/1'}
tree = ET.parse('Utsm.gpx')
root = tree.getroot()
pts = []
for trkseg in root.findall('gpx:trk/gpx:trkseg', ns):
    for trkpt in trkseg.findall('gpx:trkpt', ns):
        t = trkpt.find('gpx:time', ns)
        pts.append(t.text if t is not None and t.text else None)
print('gpx points', len(pts), 'nulls', sum(1 for p in pts if p is None))
print('first 10 times', pts[:10])
