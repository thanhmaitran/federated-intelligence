import { Link } from "react-router-dom";
import { api } from "../api";
import { Card, Page, Stat, Status } from "../components/ui";
import { fmt, fmtBytes, useAsync } from "../hooks";

export default function Overview() {
  const ov = useAsync(api.overview, []);
  const groups = useAsync(api.groups, []);
  const o = ov.data;
  return (
    <Page title="Overview" sub="Every figure on this dashboard is read from experiment runs stored by the backend; nothing is mocked.">
      <Status loading={ov.loading} error={ov.error} />
      {o && (
        <>
          <div className="grid g4">
            <Stat label="Experiment runs" value={o.num_experiments} />
            <Stat label="Best test accuracy" value={fmt(o.best_test_accuracy, 4)} />
            <Stat label="Bytes exchanged (all runs)" value={fmtBytes(o.total_bytes_transferred)} />
            <Stat label="Datasets registered" value={o.datasets} />
          </div>
          <div style={{ height: 16 }} />
          <div className="grid g2">
            <Card title="Runs by status" desc="Lifecycle of every stored run.">
              <table><tbody>{Object.entries(o.by_status).map(([k, v]) => (
                <tr key={k}><td><span className={`pill ${k}`}>{k}</span></td><td className="num">{v}</td></tr>))}</tbody></table>
            </Card>
            <Card title="Runs by experiment group" desc="Experiment matrix identifiers (EXP-xx).">
              <Status loading={groups.loading} error={groups.error} />
              <table><tbody>{(groups.data ?? []).map((g) => (
                <tr key={g.group}><td><Link className="link" to={`/history?group=${g.group}`}>{g.group}</Link></td><td className="num">{g.runs}</td></tr>))}</tbody></table>
            </Card>
          </div>
        </>
      )}
    </Page>
  );
}
