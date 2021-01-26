import argparse
import os
from pprint import pprint

import couchdb


class ServerWithCompletion(couchdb.Server):
    def _ipython_key_completions_(self):
        return list(self)


def index_exists(db, ddoc, name):
    return any(
        idx['ddoc'] == '_design/{}'.format(ddoc) and idx['name'] == name
        for idx in db.index()
    )


DB_CONFIG = os.path.join(os.path.dirname(__file__), 'database.txt')

if not os.path.exists(DB_CONFIG):
    print('Missing {}, please provide file, e.g.:'.format(DB_CONFIG))
    print(open(DB_CONFIG+'.example').read())
    srv = db = None
else:
    server_url, db_name = open(DB_CONFIG).read().strip().split("\n")

    srv = ServerWithCompletion(server_url)

    if db_name not in srv:
        print("{} not on the server. Creating.".format(db_name))
        db = srv.create(db_name)
    else:
        db = srv[db_name]

    # indexes
    idx = db.index()

    if not index_exists(db, 'main', 'type'):
        print("Creating 'type' index...")
        idx['main', 'type'] = ['type']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--push', help='remote database to push to')
    parser.add_argument('--pull', help='remote database to pull from')
    args = parser.parse_args()

    if args.push:
        result = srv.replicate(db_name, args.push)
        pprint(result)
    if args.pull:
        result = srv.replicate(args.pull, db_name)
        pprint(result)
