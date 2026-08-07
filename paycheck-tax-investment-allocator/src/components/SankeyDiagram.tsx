import React, { useMemo, useState, useRef, useEffect } from 'react';
import { sankey, sankeyLinkHorizontal, SankeyNode, SankeyLink } from 'd3-sankey';
import { UserFinancialInputs, TaxBreakdownResult } from '../types';
import { generateSankeyData } from '../utils/taxCalculator';
import { motion, AnimatePresence } from 'motion/react';
import { Layers, Eye, Info, Sparkles, SplitSquareVertical, ZoomIn, DollarSign, Maximize2, Minimize2 } from 'lucide-react';

interface SankeyDiagramProps {
  inputs: UserFinancialInputs;
  taxResult: TaxBreakdownResult;
  onToggleDissectTaxes: () => void;
  onToggleIncludePostTax?: () => void;
  isFocusMode?: boolean;
  onToggleFocusMode?: () => void;
  chartRef?: React.RefObject<HTMLDivElement | null>;
}

interface CustomNode {
  id: string;
  name: string;
  category: string;
  valueBiweekly: number;
  valueAnnual: number;
  percentageOfGross: number;
  color?: string;
  index?: number;
  x0?: number;
  x1?: number;
  y0?: number;
  y1?: number;
}

interface CustomLink {
  source: CustomNode | number | string;
  target: CustomNode | number | string;
  value: number;
  formattedValue: string;
  percentage: number;
  color?: string;
  width?: number;
  y0?: number;
  y1?: number;
}

export const SankeyDiagram: React.FC<SankeyDiagramProps> = ({
  inputs,
  taxResult,
  onToggleDissectTaxes,
  onToggleIncludePostTax,
  isFocusMode,
  onToggleFocusMode,
  chartRef,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerWidth, setContainerWidth] = useState<number>(850);
  const [hoveredNode, setHoveredNode] = useState<CustomNode | null>(null);
  const [hoveredLink, setHoveredLink] = useState<CustomLink | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Attach external ref if provided
  useEffect(() => {
    if (chartRef && containerRef.current) {
      (chartRef as React.MutableRefObject<HTMLDivElement | null>).current = containerRef.current;
    }
  }, [chartRef]);

  // Handle container resize cleanly
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      if (entries[0] && entries[0].contentRect.width > 0) {
        setContainerWidth(Math.max(450, entries[0].contentRect.width));
      }
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  // Layout calculations with d3-sankey
  const { nodes, links, height } = useMemo(() => {
    const raw = generateSankeyData(inputs, taxResult);
    if (!raw.nodes.length || !raw.links.length) {
      return { nodes: [], links: [], height: 680 };
    }

    const nodeCount = raw.nodes.length;
    // Calculate generous dynamic height to eliminate vertical label smudging
    const calcHeight = Math.max(700, nodeCount * 55);

    // Clone deep to avoid d3 mutating original
    const nodesCopy: CustomNode[] = raw.nodes.map((n) => ({ ...n }));
    const linksCopy: CustomLink[] = raw.links.map((l) => ({ ...l }));

    const nodeMap = new Map<string, number>();
    nodesCopy.forEach((node, idx) => {
      nodeMap.set(node.id, idx);
    });

    // Map source/target strings to indices
    const mappedLinks = linksCopy
      .map((l) => {
        const sourceIdx = typeof l.source === 'string' ? nodeMap.get(l.source) : l.source;
        const targetIdx = typeof l.target === 'string' ? nodeMap.get(l.target) : l.target;
        if (sourceIdx === undefined || targetIdx === undefined) return null;
        return {
          ...l,
          source: sourceIdx,
          target: targetIdx,
        };
      })
      .filter((l): l is NonNullable<typeof l> => l !== null) as unknown as CustomLink[];

    const margin = { top: 30, right: 185, bottom: 30, left: 180 };

    const sankeyGenerator = sankey<CustomNode, CustomLink>()
      .nodeWidth(22)
      .nodePadding(28)
      .extent([
        [margin.left, margin.top],
        [containerWidth - margin.right, calcHeight - margin.bottom],
      ]);

    try {
      const graph = sankeyGenerator({
        nodes: nodesCopy,
        links: mappedLinks,
      });
      return { nodes: graph.nodes, links: graph.links, height: calcHeight };
    } catch (e) {
      console.warn('Sankey layout error:', e);
      return { nodes: [], links: [], height: calcHeight };
    }
  }, [inputs, taxResult, containerWidth]);

  const isBiweekly = inputs.payFrequency !== 'annual';
  const fmt = (val: number) =>
    new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(val);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    setMousePos({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top,
    });
  };

  return (
    <div
      ref={containerRef}
      onMouseMove={handleMouseMove}
      className={`relative bg-white border border-slate-200 rounded-2xl p-5 text-slate-900 overflow-hidden transition-all duration-300 ${
        isFocusMode ? 'shadow-2xl ring-2 ring-indigo-500/50' : 'shadow-xs'
      }`}
    >
      {/* Top Banner / Toolbar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4 pb-3 border-b border-slate-100">
        <div>
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-[#1ab394]" />
            <h2 className="text-lg font-bold tracking-tight text-slate-900">
              Paycheck Sankey Flow
            </h2>
            <span className="px-2 py-0.5 text-xs font-semibold rounded-full bg-[#1ab394]/10 text-[#1ab394] border border-[#1ab394]/30">
              {isBiweekly ? 'Biweekly' : 'Annual'}
            </span>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Visualize how gross paycheck converts into taxes, investments, and net cash. Click <span className="text-rose-500 font-medium">Taxes</span> to dissect!
          </p>
        </div>

        <div className="flex items-center gap-2">
          {onToggleFocusMode && (
            <button
              onClick={onToggleFocusMode}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-bold rounded-lg transition-all border shadow-xs ${
                isFocusMode
                  ? 'bg-indigo-600 text-white border-indigo-500 hover:bg-indigo-700'
                  : 'bg-slate-100 text-slate-700 border-slate-200 hover:bg-slate-200'
              }`}
              title={isFocusMode ? 'Exit Chart Focus Mode' : 'Focus Chart View'}
            >
              {isFocusMode ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
              <span>{isFocusMode ? 'Normal View' : 'Focus Chart'}</span>
            </button>
          )}

          {onToggleIncludePostTax && (
            <button
              onClick={onToggleIncludePostTax}
              className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition-all border shadow-xs ${
                inputs.includePostTaxInSankey
                  ? 'bg-emerald-600 text-white border-emerald-500 hover:bg-emerald-700'
                  : 'bg-slate-100 text-slate-700 border-slate-200 hover:bg-slate-200'
              }`}
              title="Option to include post-tax investments (Roth, 529, Child, ESPP) in Sankey flow"
            >
              <Eye className="w-3.5 h-3.5" />
              <span>{inputs.includePostTaxInSankey ? 'Hide Post-Tax Allocations' : '+ Add Post-Tax Accounts (Roth/529/Child/ESPP)'}</span>
            </button>
          )}

          <button
            onClick={onToggleDissectTaxes}
            id="btn-dissect-taxes"
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold rounded-lg transition-all border shadow-xs ${
              inputs.dissectTaxesInSankey
                ? 'bg-rose-50 text-rose-700 border-rose-200 hover:bg-rose-100'
                : 'bg-slate-800 text-slate-100 border-slate-700 hover:bg-slate-900'
            }`}
          >
            <SplitSquareVertical className="w-3.5 h-3.5" />
            {inputs.dissectTaxesInSankey ? 'Collapse Taxes' : 'Dissect Taxes (Detailed)'}
          </button>
        </div>
      </div>

      {/* Sankey SVG Container */}
      <div className="relative w-full overflow-x-auto min-h-[460px] flex justify-center items-center">
        {nodes.length === 0 ? (
          <div className="text-center py-12 text-slate-400 text-sm">
            Please enter a gross salary number above to generate your paycheck Sankey flow.
          </div>
        ) : (
          <svg
            width={containerWidth}
            height={height}
            className="w-full h-auto select-none overflow-visible"
          >
            <defs>
              {/* Linear Gradients for links */}
              {links.map((link, idx) => {
                const sourceNode = link.source as CustomNode;
                const targetNode = link.target as CustomNode;
                const gradientId = `sankey-gradient-${idx}`;
                return (
                  <linearGradient
                    key={gradientId}
                    id={gradientId}
                    gradientUnits="userSpaceOnUse"
                    x1={sourceNode.x1}
                    x2={targetNode.x0}
                  >
                    <stop offset="0%" stopColor={sourceNode.color || '#6366f1'} stopOpacity={0.45} />
                    <stop offset="100%" stopColor={targetNode.color || '#10b981'} stopOpacity={0.45} />
                  </linearGradient>
                );
              })}
            </defs>

            {/* Links Path */}
            <g className="links">
              {links.map((link, idx) => {
                const linkPath = sankeyLinkHorizontal()(link as unknown as SankeyLink<CustomNode, CustomLink>);
                const isHovered = hoveredLink === link;
                return (
                  <path
                    key={`link-${idx}`}
                    d={linkPath || ''}
                    fill="none"
                    stroke={`url(#sankey-gradient-${idx})`}
                    strokeWidth={Math.max(2, link.width || 0)}
                    strokeOpacity={isHovered ? 0.85 : 0.45}
                    className="transition-all duration-200 cursor-pointer hover:stroke-opacity-90"
                    onMouseEnter={() => setHoveredLink(link)}
                    onMouseLeave={() => setHoveredLink(null)}
                  />
                );
              })}
            </g>

            {/* Nodes */}
            <g className="nodes">
              {nodes.map((node, idx) => {
                const x0 = node.x0 ?? 0;
                const x1 = node.x1 ?? 0;
                const y0 = node.y0 ?? 0;
                const y1 = node.y1 ?? 0;
                const nodeHeight = Math.max(8, y1 - y0);
                const isHovered = hoveredNode === node;
                const isTaxNode = node.id === 'taxes' || node.category === 'taxChild';

                return (
                  <g
                    key={`node-${idx}`}
                    className="cursor-pointer group"
                    onMouseEnter={() => setHoveredNode(node)}
                    onMouseLeave={() => setHoveredNode(null)}
                    onClick={() => {
                      if (node.id === 'taxes' || node.category === 'taxChild') {
                        onToggleDissectTaxes();
                      }
                    }}
                  >
                    {/* Node Bar */}
                    <rect
                      x={x0}
                      y={y0}
                      width={x1 - x0}
                      height={nodeHeight}
                      fill={node.color || '#6366f1'}
                      rx={4}
                      className={`transition-all duration-200 ${
                        isHovered ? 'brightness-110 filter stroke-2 stroke-white' : ''
                      }`}
                    />

                    {/* Node Label Text - High Contrast */}
                    <text
                      x={x0 < containerWidth / 2 ? x1 + 8 : x0 - 8}
                      y={y0 + nodeHeight / 2}
                      dy="0.35em"
                      textAnchor={x0 < containerWidth / 2 ? 'start' : 'end'}
                      stroke="#ffffff"
                      strokeWidth="3.5px"
                      paintOrder="stroke fill"
                      strokeLinejoin="round"
                      className={`text-[12px] font-bold transition-all duration-200 ${
                        isHovered ? 'fill-indigo-600' : 'fill-slate-900'
                      }`}
                    >
                      {node.name}
                    </text>

                    {/* Value Badge below/next to label - High Contrast */}
                    <text
                      x={x0 < containerWidth / 2 ? x1 + 8 : x0 - 8}
                      y={y0 + nodeHeight / 2 + 15}
                      dy="0.35em"
                      textAnchor={x0 < containerWidth / 2 ? 'start' : 'end'}
                      stroke="#ffffff"
                      strokeWidth="3px"
                      paintOrder="stroke fill"
                      strokeLinejoin="round"
                      className="text-[11px] font-mono font-semibold fill-slate-700"
                    >
                      {fmt(isBiweekly ? node.valueBiweekly : node.valueAnnual)} (
                      {node.percentageOfGross.toFixed(1)}%)
                    </text>
                  </g>
                );
              })}
            </g>
          </svg>
        )}
      </div>

      {/* Floating Hover Tooltip */}
      <AnimatePresence>
        {hoveredNode && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.1 }}
            style={{
              position: 'absolute',
              left: Math.min(mousePos.x + 15, containerWidth - 220),
              top: Math.max(10, mousePos.y - 80),
              pointerEvents: 'none',
            }}
            className="z-50 bg-slate-950/95 border border-slate-700 shadow-xl rounded-xl p-3 text-xs w-56 backdrop-blur-md"
          >
            <div className="flex items-center gap-1.5 font-bold text-slate-100 mb-1">
              <span
                className="w-2.5 h-2.5 rounded-full inline-block"
                style={{ backgroundColor: hoveredNode.color || '#6366f1' }}
              />
              {hoveredNode.name}
            </div>

            <div className="space-y-1 text-slate-300 font-mono">
              <div className="flex justify-between">
                <span className="text-slate-400">Biweekly:</span>
                <span className="font-semibold text-emerald-400">{fmt(hoveredNode.valueBiweekly)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">Annualized:</span>
                <span className="font-semibold text-white">{fmt(hoveredNode.valueAnnual)}</span>
              </div>
              <div className="flex justify-between border-t border-slate-800 pt-1 mt-1 text-[11px]">
                <span className="text-slate-400">% of Gross Pay:</span>
                <span className="font-bold text-indigo-300">{hoveredNode.percentageOfGross.toFixed(1)}%</span>
              </div>
            </div>

            {(hoveredNode.id === 'taxes' || hoveredNode.category === 'taxChild') && (
              <div className="mt-2 pt-1.5 border-t border-slate-800 text-[10px] text-rose-300 flex items-center gap-1">
                <Info className="w-3 h-3 text-rose-400" />
                Click to toggle detailed state/federal tax breakdown!
              </div>
            )}
          </motion.div>
        )}

        {hoveredLink && !hoveredNode && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.1 }}
            style={{
              position: 'absolute',
              left: Math.min(mousePos.x + 15, containerWidth - 200),
              top: Math.max(10, mousePos.y - 60),
              pointerEvents: 'none',
            }}
            className="z-50 bg-slate-950/95 border border-indigo-900/60 shadow-xl rounded-xl p-2.5 text-xs w-48 backdrop-blur-md"
          >
            <div className="font-bold text-slate-200">
              {(hoveredLink.source as CustomNode).name} &rarr; {(hoveredLink.target as CustomNode).name}
            </div>
            <div className="mt-1 font-mono text-emerald-400 font-semibold">
              {hoveredLink.formattedValue}
            </div>
            <div className="text-[10px] text-slate-400">
              {hoveredLink.percentage.toFixed(1)}% of gross paycheck
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Legend Footer */}
      <div className="mt-2 pt-3 border-t border-slate-800 flex flex-wrap items-center justify-between gap-3 text-xs text-slate-400">
        <div className="flex flex-wrap items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-blue-500 inline-block" />
            <span>Gross Pay</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-purple-500 inline-block" />
            <span>Pre-Tax (401k/HSA)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-rose-500 inline-block" />
            <span>Taxes (Fed/State/FICA/SDI)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-emerald-500 inline-block" />
            <span>Post-Tax / Child (Roth/529/ESPP)</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-3 h-3 rounded bg-green-600 inline-block" />
            <span>Net Take-Home Pay</span>
          </div>
        </div>

        <div className="text-[11px] text-slate-500 italic">
          Hover over nodes or links for detailed metrics.
        </div>
      </div>
    </div>
  );
};
