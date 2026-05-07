export function formatClassificationLabel(type: string, compact = false) {
  const verboseMap: Record<string, string> = {
    FRAUD: 'FRAUD',
    RARITY: 'BENEFICIAL RARITY',
    HONEST: 'HONEST',
    NOISE: 'NOISE',
  }

  const compactMap: Record<string, string> = {
    FRAUD: 'FRAUD',
    RARITY: 'RARE+',
    HONEST: 'HONEST',
    NOISE: 'NOISE',
  }

  return compact ? (compactMap[type] || type) : (verboseMap[type] || type)
}
