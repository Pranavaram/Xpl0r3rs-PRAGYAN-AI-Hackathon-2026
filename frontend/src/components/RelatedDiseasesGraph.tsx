import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import {
  relatedDiseasesGraphApi,
  type RelatedDiseasesGraphResponse,
  type RelatedDiseasesGraphNode,
  type RelatedDiseasesGraphEdge,
} from '../api/client';
import { useLanguage } from '../context/LanguageContext';

const W = 560;
const H = 340;
const NODE_R = 20;
const PAD = 40;

function layoutNodes(nodes: RelatedDiseasesGraphNode[]): Map<string, { x: number; y: number }> {
  const positions = new Map<string, { x: number; y: number }>();
  const conditions = nodes.filter((n) => n.type === 'condition');
  const symptoms = nodes.filter((n) => n.type === 'symptom');
  const leftX = PAD + (W / 2 - PAD * 2) * 0.35;
  const rightX = W - PAD - (W / 2 - PAD * 2) * 0.35;
  const vertPad = (H - 2 * PAD) / Math.max(conditions.length, 1);
  conditions.forEach((n, i) => {
    positions.set(n.id, { x: leftX, y: PAD + (i + 0.5) * vertPad });
  });
  const vertPadS = (H - 2 * PAD) / Math.max(symptoms.length, 1);
  symptoms.forEach((n, i) => {
    positions.set(n.id, { x: rightX, y: PAD + (i + 0.5) * vertPadS });
  });
  return positions;
}

export function RelatedDiseasesGraph() {
  const { t } = useLanguage();
  const [data, setData] = useState<RelatedDiseasesGraphResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [hoveredEdge, setHoveredEdge] = useState<RelatedDiseasesGraphEdge | null>(null);

  useEffect(() => {
    let cancelled = false;
    relatedDiseasesGraphApi()
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : 'Failed to load graph');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => { cancelled = true; };
  }, []);

  if (loading) {
    return (
      <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-6 bg-white dark:bg-gray-800 min-h-[340px] flex items-center justify-center">
        <p className="text-gray-500">{t('resultsRelatedDiseasesLoading')}</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-6 bg-white dark:bg-gray-800 min-h-[340px] flex items-center justify-center">
        <p className="text-gray-500">{t('resultsRelatedDiseasesError')}</p>
      </div>
    );
  }

  const positions = layoutNodes(data.nodes);
  const nodeList = data.nodes;
  const edgeList = data.edges;

  return (
    <div className="rounded-xl border border-gray-200 dark:border-gray-700 p-6 bg-white dark:bg-gray-800">
      <h3 className="font-semibold mb-2">{t('resultsRelatedDiseasesGraph')}</h3>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-4">
        {t('resultsRelatedDiseasesHint')}
      </p>
      <div className="relative">
        <svg
          width="100%"
          height={H}
          viewBox={`0 0 ${W} ${H}`}
          className="overflow-visible"
          style={{ maxWidth: W }}
        >
          <defs>
            <marker
              id="arrow"
              markerWidth="8"
              markerHeight="8"
              refX="6"
              refY="4"
              orient="auto"
              markerUnits="strokeWidth"
            >
              <path d="M0,0 L0,8 L8,4 z" fill="currentColor" className="text-gray-400" />
            </marker>
          </defs>
          {/* edges */}
          {edgeList.map((edge, i) => {
            const src = positions.get(edge.source);
            const tgt = positions.get(edge.target);
            if (!src || !tgt) return null;
            const isHovered = hoveredEdge === edge;
            return (
              <g
                key={`${edge.source}-${edge.target}-${i}`}
                onMouseEnter={() => setHoveredEdge(edge)}
                onMouseLeave={() => setHoveredEdge(null)}
              >
                {/* Invisible wide stroke for easier hover */}
                <line
                  x1={src.x}
                  y1={src.y}
                  x2={tgt.x}
                  y2={tgt.y}
                  stroke="transparent"
                  strokeWidth={14}
                />
                <line
                  x1={src.x}
                  y1={src.y}
                  x2={tgt.x}
                  y2={tgt.y}
                  stroke={isHovered ? '#0d9488' : '#94a3b8'}
                  strokeWidth={isHovered ? 2.5 : 1.2}
                  strokeDasharray={isHovered ? undefined : '4 3'}
                  markerEnd="url(#arrow)"
                />
              </g>
            );
          })}
          {/* nodes */}
          {nodeList.map((node) => {
            const pos = positions.get(node.id);
            if (!pos) return null;
            const isCondition = node.type === 'condition';
            return (
              <g key={node.id}>
                <motion.circle
                  r={NODE_R}
                  cx={pos.x}
                  cy={pos.y}
                  fill={isCondition ? '#1e3a8a' : '#0d9488'}
                  stroke="#fff"
                  strokeWidth="2"
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.05 * nodeList.indexOf(node), type: 'spring', stiffness: 200 }}
                />
                <text
                  x={pos.x + (isCondition ? -NODE_R - 4 : NODE_R + 4)}
                  y={pos.y}
                  textAnchor={isCondition ? 'end' : 'start'}
                  dominantBaseline="middle"
                  className="text-xs font-medium fill-gray-700 dark:fill-gray-200"
                >
                  {node.label}
                </text>
              </g>
            );
          })}
        </svg>
        {hoveredEdge && (
          <div
            className="absolute bottom-0 left-0 right-0 p-3 rounded-lg bg-teal-50 dark:bg-gray-700 border border-teal-200 dark:border-gray-600 text-sm"
            role="tooltip"
          >
            <span className="font-medium text-teal-800 dark:text-teal-200">
              +{hoveredEdge.weight} risk
              {hoveredEdge.preferred_department ? ` → ${hoveredEdge.preferred_department}` : ''}
            </span>
            <p className="text-gray-600 dark:text-gray-300 mt-1">{hoveredEdge.explanation}</p>
          </div>
        )}
      </div>
    </div>
  );
}
