import json
import pprint
import datetime as dt
import pvlib.solarposition as sp

def prettyPrint(data, indent=4):
	pp = pprint.PrettyPrinter(indent)

	pp.pprint(data)


def get_distance(waggles, waggle):


	dist = (float(waggles[waggle]['final_frame']) - float(waggles[waggle]['init_frame'])) / float(waggles['FPS'])

def detectLocation(pickleLocation, filePrefix):
	with open(f'{pickleLocation}/{filePrefix}-danceData.json', 'r') as f:
		waggles = json.load(f)

	date = dt.datetime.strptime(waggles['recordDate'], '%a %b %d %H:%M:%S %Y')
	lat, lon = (37.561343, -77.465806)

	solarposition = sp.get_solarposition(date, lat, lon)

	waggles['azimuth'] = str(solarposition['azimuth'][0])


	keys = [k for k, v in waggles.items() if 'waggle-' in k]


	for waggle in keys:
		waggles[waggle]['distance'] = get_distance(waggles, waggle)
	

	prettyPrint(waggles)

	with open(f'{pickleLocation}/{filePrefix}-danceData.json', 'w') as f:
		json.dump(waggles, f)

if __name__ == '__main__':
    detectLocation('../DB/danceData', 'bee-cropped-0')
