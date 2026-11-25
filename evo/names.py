"""Stable, readable display names. Technical IDs remain the storage keys."""
import hashlib
import re

GIVEN = '''Atlas Nova Orion Sage Echo Luna Vega Scout Milo Finn Theo Ava Ivy Leo Ruby Arlo
Nora Max Isla Owen Zoe Ezra Maya Kai Ada Eli Remy Juno Cleo Hugo Luca Iris
Oscar Wren Felix Olive Jasper Hazel Rowan Piper Ellis Harper River Sasha Reese Toby Mira Quinn
Robin Avery Eden Jude Skye Rory Blair Drew Indie Jamie Casey Ari Blake Morgan Riley Emery'''.split()
MIDDLE = '''Cedar Willow Maple Birch Aspen Alder Laurel Linden Juniper Cypress Hawthorn Oak Pine Elm Spruce Fir
Fern Moss Clover Meadow Brook Ocean Harbor Bay Cove Coral Pearl Amber Jade Flint Slate Jasper
Quartz Silver Copper Golden Scarlet Azure Indigo Violet Rose Iris Lily Aster Dahlia Poppy Petal Leaf
Grove Orchard Summit Ridge Valley Canyon Prairie Dune Forest Timber Sequoia Redwood Acacia Olive Hazel Rowan'''.split()
FAMILY = '''Brooks Carter Hayes Reed Bennett Parker Morgan Ellis Foster Harper Cooper Mason Turner Walker Taylor Collins
Wells Bailey Miller Davis Wilson Anderson Thomas Moore Martin Clark Lewis Robinson Hall Young Allen King
Wright Scott Green Baker Adams Nelson Hill Campbell Mitchell Roberts Phillips Evans Edwards Stewart Morris Rogers
Cook Ward Bell Murphy Russell Rivera Kelly Howard Cox Richardson Wood Watson Price Patel Sanders Perry
Powell Butler Barnes Ross Henderson Coleman Jenkins Fisher Graham Murray Gibson Wallace West Cole Jordan Owens
Reynolds Palmer Holmes Spencer Gardner Payne Grant Knight Webb Duncan Hudson Hart Lane Armstrong Austin Stone
Chapman Lawson Warren Weaver Riley Porter Hunter Hicks Crawford Henry Boyd Shaw Gordon Bishop Greene Fields
Day Daniels Cross Ford Franklin Fox Freeman Frost Gray Griffin Hale Hanson Hardy Heath Holland Hope'''.split()


def bot_name(bot):
    if bot=='buy-and-hold':
        return 'Buy and hold'
    match=re.fullmatch(r'g(\d+)-b(\d+)',str(bot))
    if match and int(match[1])<1000 and int(match[2])<500:
        # A permutation gives every supported lineage a distinct name, without a registry.
        number=(int(match[1])*500+int(match[2]))*2654435761+51
    else:
        number=int(hashlib.sha256(str(bot).encode()).hexdigest()[:12],16)
    number&=(1<<19)-1
    return f'{GIVEN[number&63]} {MIDDLE[(number>>6)&63]} {FAMILY[(number>>12)&127]}'
