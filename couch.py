import os

import couchdb


class ServerWithCompletion(couchdb.Server):
    def _ipython_key_completions_(self):
        return list(self)


DB_CONFIG = os.path.join(os.path.dirname(__file__), 'database.txt')

if not os.path.exists(DB_CONFIG):
    print(f'Missing {DB_CONFIG}, please provide file, e.g.:')
    print(open(DB_CONFIG+'.example').read())
    srv = db = None
else:
    server_url, db_name = open(DB_CONFIG).read().strip().split("\n")

    srv = ServerWithCompletion(server_url)

    if db_name not in srv:
        print(f"{db_name} not on the server. Creating.")
        db = srv.create(db_name)
    else:
        db = srv[db_name]
