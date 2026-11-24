# BSD 3-Clause License; see https://github.com/scikit-hep/uproot/blob/main/LICENSE

import datetime
import http.client
import json
import math
import re
import subprocess

from urllib.request import urlopen

tagslist_text = subprocess.run(
    ["git", "show-ref", "--tags"], stdout=subprocess.PIPE
).stdout
tagslist = {
    k: v
    for k, v in re.findall(rb"([0-9a-f]{40}) refs/tags/(v?[0-9\.rc]+)", tagslist_text)
    if not v.startswith(b"0.")
}

subjects_text = subprocess.run(
    ["git", "log", "--format='format:%H %s'"], stdout=subprocess.PIPE
).stdout
subjects = re.findall(rb"([0-9a-f]{40}) (.*)", subjects_text)

github_releases = []
numpages = int(math.ceil(len(tagslist) / 30.0))
for pageid in range(numpages):
    print(f"Requesting GitHub data, page {pageid + 1} of {numpages}")
    github_releases_text = urlopen(
        rf"https://api.github.com/repos/scikit-hep/uproot4/releases?page={pageid + 1}&per_page=30"
    ).read()
    try:
        github_releases_page = json.loads(github_releases_text)
    except:
        print(github_releases_text)
        raise
    print(len(github_releases_page))
    github_releases.extend(github_releases_page)

releases = {
    x["tag_name"]: x["body"] for x in github_releases if "tag_name" in x and "body" in x
}
dates = {
    x["tag_name"]: datetime.datetime.fromisoformat(
        x["published_at"].rstrip("Z")
    ).strftime("%A, %d %B, %Y")
    for x in github_releases
    if "tag_name" in x and "published_at" in x
}
tarballs = {
    x["tag_name"]: x["tarball_url"]
    for x in github_releases
    if "tag_name" in x and "tarball_url" in x
}
zipballs = {
    x["tag_name"]: x["zipball_url"]
    for x in github_releases
    if "tag_name" in x and "zipball_url" in x
}

pypi_connection = http.client.HTTPSConnection("pypi.org")


def pypi_exists(tag):
    print(f"Looking for release {tag} on PyPI...")
    pypi_connection.request("HEAD", f"/project/uproot/{tag}/")
    response = pypi_connection.getresponse()
    response.read()
    return response.status == 200


with open("changelog.rst", "w") as outfile:
    outfile.write("Release history\n")
    outfile.write("---------------\n")
    outfile.write(
        "Note: Releases in the 4.3.x series were developed in parallel with v5.0 on a separate branch and are not included here. See the `list of 4.x releases <https://github.com/scikit-hep/uproot/releases?q=%224.3%22+OR+%22v4.3%22&expanded=true>`__ for details.\n"
    )

    first = True
    numprs = None

    for taghash, subject in subjects:
        if taghash in tagslist:
            tag = tagslist[taghash].decode()
            tagurl = f"https://github.com/scikit-hep/uproot/releases/tag/{tag}"

            if numprs == 0:
                outfile.write("*(no pull requests)*\n")
            numprs = 0

            header_text = f"\nRelease `{tag} <{tagurl}>`__\n"
            outfile.write(header_text)
            outfile.write("=" * len(header_text) + "\n\n")

            if tag in dates:
                date_text = "**" + dates[tag] + "**"
            else:
                date_text = ""

            assets = []
            if pypi_exists(tag):
                assets.append(f"`pip <https://pypi.org/project/uproot/{tag}/>`__")
            if tag in tarballs:
                assets.append(f"`tar <{tarballs[tag]}>`__")
            if tag in zipballs:
                assets.append(f"`zip <{zipballs[tag]}>`__")
            if len(assets) == 0:
                assets_text = ""
            else:
                assets_text = " ({})".format(", ".join(assets))

            if len(date_text) + len(assets_text) > 0:
                outfile.write(f"{date_text}{assets_text}\n\n")

            if tag in releases:
                text = (
                    releases[tag]
                    .strip()
                    .replace(
                        "For details, see the [release history](https://uproot.readthedocs.io/en/latest/changelog.html).",
                        "",
                    )
                )
                text = re.sub(
                    r"([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+)#([1-9][0-9]*)",
                    r"`\1#\2 <https://github.com/\1/issues/\2>`__",
                    text,
                )
                text = re.sub(
                    r"([^a-zA-Z0-9\-_])#([1-9][0-9]*)",
                    r"\1`#\2 <https://github.com/scikit-hep/uproot/issues/\2>`__",
                    text,
                )
                outfile.write(text + "\n\n")

            first = False

        m = re.match(rb"(.*) \(#([1-9][0-9]*)\)", subject)
        if m is not None:
            if numprs is None:
                numprs = 0
            numprs += 1

            if first:
                header_text = "\nUnreleased (`main branch <https://github.com/scikit-hep/uproot>`__ on GitHub)\n"
                outfile.write(header_text)
                outfile.write("=" * len(header_text) + "\n\n")

            text = m.group(1).decode().strip()
            prnum = m.group(2).decode()
            prurl = f"https://github.com/scikit-hep/uproot/pull/{prnum}"

            known = [prnum]
            for issue in re.findall(
                r"([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+)#([1-9][0-9]*)", text
            ):
                known.append(issue)
            for issue in re.findall(r"([^a-zA-Z0-9\-_])#([1-9][0-9]*)", text):
                known.append(issue[1])

            text = re.sub(r"`", "``", text)
            text = re.sub(
                r"([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+)#([1-9][0-9]*)",
                r"`\1#\2 <https://github.com/\1/issues/\2>`__",
                text,
            )
            text = re.sub(
                r"([^a-zA-Z0-9\-_])#([1-9][0-9]*)",
                r"\1`#\2 <https://github.com/scikit-hep/uproot/issues/\2>`__",
                text,
            )
            if re.match(r".*[a-zA-Z0-9_]$", text):
                text = text + "."

            body_text = subprocess.run(
                ["git", "log", "-1", taghash.decode(), "--format='format:%b'"],
                stdout=subprocess.PIPE,
            ).stdout.decode()
            addresses = []
            for issue in re.findall(
                r"([a-zA-Z0-9\-_]+/[a-zA-Z0-9\-_]+)#([1-9][0-9]*)", body_text
            ):
                if issue not in known:
                    addresses.append(
                        "`{0}#{1} <https://github.com/{0}/issues/{1}>`__".format(
                            issue[0], issue[1]
                        )
                    )
            for issue in re.findall(r"([^a-zA-Z0-9\-_])#([1-9][0-9]*)", body_text):
                if issue[1] not in known:
                    addresses.append(
                        "`#{0} <https://github.com/scikit-hep/uproot/issues/{0}>`__".format(
                            issue[1]
                        )
                    )
            if len(addresses) == 0:
                addresses_text = ""
            else:
                addresses_text = " (**also:** {})".format(", ".join(addresses))

            outfile.write(f"* PR `#{prnum} <{prurl}>`__: {text}{addresses_text}\n")

            first = False

    outfile.write("""
Earlier releases
================

Uproot versions 1 through 3 were in a different GitHub repository: `scikit-hep/uproot3 <https://github.com/scikit-hep/uproot3>`__.

* `GitHub release notes <https://github.com/scikit-hep/uproot3/releases>`__
* `PyPI full history <https://pypi.org/project/uproot/#history>`__ (including versions 1 through 3).

This was to allow users to transition from Awkward Array 0.x and Uproot 3.x, which had different interfaces (especially Awkward Array). The transition completed on December 1, 2020.

.. image:: https://raw.githubusercontent.com/scikit-hep/uproot/main/docs-img/diagrams/uproot-awkward-timeline.png
  :width: 100%
""")
