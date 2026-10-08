// Run once by the mongo-init service (mongosh). Idempotent: safe to re-run.

// 1. Initiate the single-node replica set rs0.
try {
  rs.status();
  print("replica set already initiated");
} catch (e) {
  rs.initiate({ _id: "rs0", members: [{ _id: 0, host: "mongod:27017" }] });
  print("replica set initiated");
}

// 2. Wait until this node is PRIMARY.
for (let i = 0; i < 60 && !db.hello().isWritablePrimary; i++) {
  sleep(1000);
}
if (!db.hello().isWritablePrimary) {
  throw new Error("mongod did not become primary");
}

// 3. Create the user mongot authenticates as (role searchCoordinator).
const admin = db.getSiblingDB("admin");
if (admin.getUser("mongotUser") === null) {
  admin.createUser({
    user: "mongotUser",
    pwd: process.env.MONGOT_PASSWORD,
    roles: ["searchCoordinator"],
  });
  print("created mongotUser");
} else {
  print("mongotUser already exists");
}
