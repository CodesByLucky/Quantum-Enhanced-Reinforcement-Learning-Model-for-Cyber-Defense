import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  LineChart, Line
} from 'recharts'
import './App.css'

const comparisonData = [
  { name: 'Accuracy', DQN: 10, QDQN: 15 },
  { name: 'F1 Score', DQN: 3.3, QDQN: 3.9 },
]

const lossData = [
  { epoch: 'Epoch 1', DQN: 0.0373, QDQN: 0.1078 },
  { epoch: 'Epoch 2', DQN: 0.0182, QDQN: 0.1047 },
]

const quantumJobs = [
  { id: 'd60ejqd7fc0s73', created: '11:46 PM', completed: '11:47 PM', qpu: 'ibm_fez', usage: '37s', circuits: 20, shots: 256 },
  { id: 'd60ejc1mvbjc7', created: '11:45 PM', completed: '11:46 PM', qpu: 'ibm_fez', usage: '33s', circuits: 20, shots: 256 },
  { id: 'd60eithmvbjc73', created: '11:44 PM', completed: '11:45 PM', qpu: 'ibm_fez', usage: '34s', circuits: 20, shots: 256 },
  { id: 'd60eifhmvbjc73', created: '11:43 PM', completed: '11:44 PM', qpu: 'ibm_fez', usage: '33s', circuits: 20, shots: 256 },
  { id: 'd60eho2bju6s7', created: '11:43 PM', completed: '11:43 PM', qpu: 'ibm_fez', usage: '38s', circuits: 20, shots: 256 },
  { id: 'd60eh9ruf71s7', created: '11:40 PM', completed: '11:41 PM', qpu: 'ibm_fez', usage: '33s', circuits: 20, shots: 256 },
  { id: 'd60egr9mvbjc7', created: '11:39 PM', completed: '11:40 PM', qpu: 'ibm_fez', usage: '33s', circuits: 20, shots: 256 },
]

const results = [
  { metric: 'Accuracy', dqn: '10.0%', qdqn: '15.0%', diff: '+50%', positive: true, desc: 'Correctly classified samples out of 20 test samples' },
  { metric: 'F1 Score', dqn: '0.033', qdqn: '0.039', diff: '+18%', positive: true, desc: 'Weighted F1 across 8 threat classes' },
  { metric: 'Training Time', dqn: '0.12s', qdqn: '380s', diff: '~3000x', positive: false, desc: 'Time to complete training' },
  { metric: 'Epochs', dqn: '2', qdqn: '2', diff: 'Same', positive: false, desc: 'Complete passes through training data' },
  { metric: 'Updates', dqn: '6', qdqn: '6', diff: 'Same', positive: false, desc: 'Total gradient updates' },
  { metric: 'Train Size', dqn: '60', qdqn: '60', diff: 'Same', positive: false, desc: 'Training samples used' },
  { metric: 'Test Size', dqn: '20', qdqn: '20', diff: 'Same', positive: false, desc: 'Test samples used' },
]

export default function App() {
  const [selectedJob, setSelectedJob] = useState(null)
  const [selectedResult, setSelectedResult] = useState(null)

  return (
    <div className="min-h-screen" style={{ background: 'var(--bg)' }}>

      {/* Header */}
      <header style={{ borderBottom: '1px solid var(--border)', padding: '24px 0' }}>
        <div className="container">
          <p style={{ fontSize: 12, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: 8 }}>
            Research Project
          </p>
          <h1 style={{ fontSize: 28, fontWeight: 600, color: 'var(--text)', marginBottom: 8 }}>
            Quantum Reinforcement Learning
          </h1>
          <p style={{ fontSize: 15, color: 'var(--text-muted)', maxWidth: 600 }}>
            Fair comparison of Classical DQN vs Quantum DQN for cybersecurity threat classification
          </p>
        </div>
      </header>

      <main className="container" style={{ paddingTop: 48, paddingBottom: 80 }}>

        {/* Stats Grid */}
        <section style={{ marginBottom: 48 }}>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16 }}>
            <StatCard label="QDQN Accuracy" value="15.0%" sub="+50% vs Classical" color="var(--accent-blue)" />
            <StatCard label="DQN Accuracy" value="10.0%" sub="Baseline model" color="var(--accent-purple)" />
            <StatCard label="Quantum Jobs" value="7" sub="IBM ibm_fez" color="var(--accent-cyan)" />
            <StatCard label="Total Runtime" value="6.3 min" sub="Under 10 min budget" color="var(--accent-green)" />
          </div>
        </section>

        {/* Charts */}
        <section style={{ marginBottom: 48 }}>
          <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 24 }}>Performance Analysis</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 24 }}>

            {/* Bar Chart */}
            <div className="card" style={{ padding: 24 }}>
              <h3 style={{ fontSize: 15, fontWeight: 500, marginBottom: 4 }}>Accuracy Comparison</h3>
              <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 24 }}>Performance on 20 test samples</p>
              <div style={{ height: 240 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={comparisonData} barGap={8}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                    <XAxis dataKey="name" stroke="#71717a" fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis stroke="#71717a" fontSize={12} tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTooltip />} />
                    <Bar dataKey="DQN" fill="#a855f7" radius={[4, 4, 0, 0]} />
                    <Bar dataKey="QDQN" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
              <div style={{ display: 'flex', justifyContent: 'center', gap: 24, marginTop: 16 }}>
                <Legend color="#a855f7" label="DQN (Classical)" />
                <Legend color="#3b82f6" label="QDQN (Quantum)" />
              </div>
            </div>

            {/* Line Chart */}
            <div className="card" style={{ padding: 24 }}>
              <h3 style={{ fontSize: 15, fontWeight: 500, marginBottom: 4 }}>Training Loss</h3>
              <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 24 }}>Loss over 2 training epochs</p>
              <div style={{ height: 240 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={lossData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                    <XAxis dataKey="epoch" stroke="#71717a" fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis stroke="#71717a" fontSize={12} tickLine={false} axisLine={false} />
                    <Tooltip content={<ChartTooltip />} />
                    <Line type="monotone" dataKey="DQN" stroke="#a855f7" strokeWidth={2} dot={{ r: 4, fill: '#a855f7' }} />
                    <Line type="monotone" dataKey="QDQN" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4, fill: '#3b82f6' }} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div style={{ display: 'flex', justifyContent: 'center', gap: 24, marginTop: 16 }}>
                <Legend color="#a855f7" label="DQN" />
                <Legend color="#3b82f6" label="QDQN" />
              </div>
            </div>
          </div>
        </section>

        {/* Results Table */}
        <section style={{ marginBottom: 48 }}>
          <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>Detailed Results</h2>
          <p style={{ fontSize: 14, color: 'var(--text-muted)', marginBottom: 24 }}>Click any row for details</p>
          <div className="card table-container">
            <table>
              <thead>
                <tr>
                  <th style={{ width: '25%' }}>Metric</th>
                  <th style={{ width: '20%' }}>DQN</th>
                  <th style={{ width: '20%' }}>QDQN</th>
                  <th style={{ width: '35%' }}>Comparison</th>
                </tr>
              </thead>
              <tbody>
                {results.map((row, i) => (
                  <tr key={i} onClick={() => setSelectedResult(row)}>
                    <td style={{ fontWeight: 500 }}>{row.metric}</td>
                    <td style={{ color: 'var(--text-muted)' }}>{row.dqn}</td>
                    <td style={{ color: 'var(--text-muted)' }}>{row.qdqn}</td>
                    <td style={{ color: row.positive ? 'var(--accent-green)' : 'var(--text-muted)' }}>{row.diff}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* Quantum Jobs */}
        <section style={{ marginBottom: 48 }}>
          <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>Quantum Execution Log</h2>
          <p style={{ fontSize: 14, color: 'var(--text-muted)', marginBottom: 24 }}>7 jobs executed on IBM ibm_fez (156 qubits)</p>
          <div className="card table-container">
            <table>
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Status</th>
                  <th>Created</th>
                  <th>Completed</th>
                  <th>QPU</th>
                  <th>Usage</th>
                </tr>
              </thead>
              <tbody>
                {quantumJobs.map((job, i) => (
                  <tr key={i} onClick={() => setSelectedJob(job)}>
                    <td style={{ fontFamily: 'monospace', fontSize: 13 }}>{job.id}</td>
                    <td><span className="status">Completed</span></td>
                    <td style={{ color: 'var(--text-muted)' }}>{job.created}</td>
                    <td style={{ color: 'var(--text-muted)' }}>{job.completed}</td>
                    <td style={{ color: 'var(--accent-cyan)' }}>{job.qpu}</td>
                    <td>{job.usage}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        {/* IBM Screenshot */}
        <section style={{ marginBottom: 48 }}>
          <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 24 }}>IBM Quantum Platform</h2>
          <div className="card" style={{ padding: 24 }}>
            <img
              src="/ibm-quantum-jobs.png"
              alt="IBM Quantum Platform"
              style={{ width: '100%', borderRadius: 8, border: '1px solid var(--border)' }}
            />
          </div>
        </section>

        {/* Methodology */}
        <section>
          <h2 style={{ fontSize: 18, fontWeight: 600, marginBottom: 24 }}>Methodology</h2>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: 24 }}>
            <div className="card" style={{ padding: 24 }}>
              <h3 style={{ fontSize: 15, fontWeight: 500, marginBottom: 16 }}>Fair Comparison Protocol</h3>
              <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                {['Same training data (60 samples)', 'Same test data (20 samples)', 'Same epochs (2)', 'Same gradient updates (6)', 'Fixed random seed (42)'].map((item, i) => (
                  <li key={i} style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12, fontSize: 14, color: 'var(--text-muted)' }}>
                    <span style={{ width: 6, height: 6, borderRadius: '50%', background: 'var(--accent-green)', flexShrink: 0 }} />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
            <div className="card" style={{ padding: 24 }}>
              <h3 style={{ fontSize: 15, fontWeight: 500, marginBottom: 16 }}>Model Architectures</h3>
              <div style={{ marginBottom: 16 }}>
                <p style={{ fontSize: 14, fontWeight: 500, color: 'var(--accent-purple)', marginBottom: 4 }}>Classical DQN</p>
                <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>13 inputs, 128x2 hidden neurons, 8 outputs (~18K params)</p>
              </div>
              <div>
                <p style={{ fontSize: 14, fontWeight: 500, color: 'var(--accent-blue)', marginBottom: 4 }}>Quantum DQN</p>
                <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>4-qubit RealAmplitudes circuit, 12 trainable rotation parameters</p>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer style={{ borderTop: '1px solid var(--border)', padding: '32px 0', textAlign: 'center' }}>
        <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>
          Quantum Reinforcement Learning Research | IBM Quantum | February 2026
        </p>
      </footer>

      {/* Job Modal */}
      <Modal isOpen={!!selectedJob} onClose={() => setSelectedJob(null)}>
        {selectedJob && (
          <>
            <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 24 }}>Job Details</h3>
            <div style={{ display: 'grid', gap: 12 }}>
              <Row label="Job ID" value={selectedJob.id} mono />
              <Row label="Status" value="Completed" status />
              <Row label="QPU" value={selectedJob.qpu} accent />
              <Row label="Created" value={selectedJob.created} />
              <Row label="Completed" value={selectedJob.completed} />
              <Row label="Usage" value={selectedJob.usage} />
              <div style={{ borderTop: '1px solid var(--border)', margin: '12px 0' }} />
              <Row label="Circuits" value={selectedJob.circuits} />
              <Row label="Shots" value={selectedJob.shots} />
            </div>
          </>
        )}
      </Modal>

      {/* Result Modal */}
      <Modal isOpen={!!selectedResult} onClose={() => setSelectedResult(null)}>
        {selectedResult && (
          <>
            <h3 style={{ fontSize: 18, fontWeight: 600, marginBottom: 8 }}>{selectedResult.metric}</h3>
            <p style={{ fontSize: 14, color: 'var(--text-muted)', marginBottom: 24 }}>{selectedResult.desc}</p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12 }}>
              <ValueCard label="DQN" value={selectedResult.dqn} color="var(--accent-purple)" />
              <ValueCard label="QDQN" value={selectedResult.qdqn} color="var(--accent-blue)" />
              <ValueCard label="Diff" value={selectedResult.diff} color={selectedResult.positive ? 'var(--accent-green)' : 'var(--text-muted)'} />
            </div>
          </>
        )}
      </Modal>
    </div>
  )
}

function StatCard({ label, value, sub, color }) {
  return (
    <div className="card" style={{ padding: 20 }}>
      <p style={{ fontSize: 11, fontWeight: 500, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 8 }}>{label}</p>
      <p style={{ fontSize: 32, fontWeight: 700, color, marginBottom: 4, lineHeight: 1 }}>{value}</p>
      <p style={{ fontSize: 13, color: 'var(--text-muted)' }}>{sub}</p>
    </div>
  )
}

function Legend({ color, label }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <span style={{ width: 12, height: 12, borderRadius: 3, background: color }} />
      <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>{label}</span>
    </div>
  )
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload) return null
  return (
    <div style={{ background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 8, padding: '12px 16px' }}>
      <p style={{ fontSize: 13, color: 'var(--text-muted)', marginBottom: 8 }}>{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{ fontSize: 13, color: entry.color, marginBottom: 2 }}>
          {entry.name}: {entry.value}
        </p>
      ))}
    </div>
  )
}

function Modal({ isOpen, onClose, children }) {
  if (!isOpen) return null
  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="modal-backdrop"
        onClick={onClose}
      />
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="modal"
      >
        <button className="modal-close" onClick={onClose}>
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none">
            <path d="M12 4L4 12M4 4L12 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
          </svg>
        </button>
        {children}
      </motion.div>
    </AnimatePresence>
  )
}

function Row({ label, value, mono, status, accent }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <span style={{ fontSize: 14, color: 'var(--text-muted)' }}>{label}</span>
      {status ? (
        <span className="status">Completed</span>
      ) : (
        <span style={{ fontSize: 14, fontFamily: mono ? 'monospace' : 'inherit', color: accent ? 'var(--accent-cyan)' : 'var(--text)' }}>{value}</span>
      )}
    </div>
  )
}

function ValueCard({ label, value, color }) {
  return (
    <div style={{ background: 'rgba(255,255,255,0.03)', borderRadius: 8, padding: 16, textAlign: 'center' }}>
      <p style={{ fontSize: 11, color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: 4 }}>{label}</p>
      <p style={{ fontSize: 20, fontWeight: 600, color }}>{value}</p>
    </div>
  )
}
