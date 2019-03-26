// Basic modules for compact routing. Generates random coordinates and levels, computes clusters

// Cristina Basescu


package main

import (
	"flag"
	"gopkg.in/dedis/onet.v2/log"
	"math"
	"math/rand"
	"strconv"
	"time"
)

var RndSrc *rand.Rand

type Node struct {
	Name string
	X int
	Y int
	Level int
	ADist []float64
	pDist []string
	cluster []string
}

// generate random coordinates for nodes
func genNodes(N int, SpaceMax int, K int) []Node{
	nodes := make([]Node, N)
	for i := 0 ; i < N ; i++ {
		nodes[i].X = RndSrc.Intn(SpaceMax)
		//nodes[i].Y = RndSrc.Intn(SpaceMax)
		nodes[i].Y = 0
		nodes[i].Name = strconv.Itoa(i)
		nodes[i].ADist = make([]float64, K)
		for j := 0 ; j < K ; j++ {
			nodes[i].ADist[j] = math.MaxInt64
		}
		nodes[i].pDist = make([]string, K)
		nodes[i].cluster = make([]string, 0)
	}



	return nodes
}

func genLevels(N int, K int, nodes []Node) {

	prob := 1.0 / math.Pow(float64(N), 1.0/float64(K))

	for lvl := 0 ; lvl < K ; lvl++ {
		for i := 0; i < N; i++ {
			if nodes[i].Level == lvl - 1 {
				rnd := RndSrc.Float64()
				if rnd < prob {
					nodes[i].Level = lvl
				}
			}
		}
	}

	for i := 0; i < N; i++ {
		log.Lvl1(nodes[i].X, nodes[i].Y, nodes[i].Level)
	}

}

func euclidianDist(a Node, b Node) float64{
	return math.Sqrt(math.Pow(float64(a.X - b.X), 2.0) + math.Pow(float64(a.Y - b.Y), 2.0))
}

func computeADist(nodes []Node, K int) {
	N := len(nodes)
	for i := 0; i < N; i++ {
		crtNode := &nodes[i]
		for lvl := K-1 ; lvl >= 1 ; lvl-- {
			for j := 0; j < N; j++ {
				dist := euclidianDist(nodes[i], nodes[j])
				if nodes[j].Level >= lvl && dist < crtNode.ADist[lvl] {
					crtNode.ADist[lvl] = dist
					crtNode.pDist[lvl] = nodes[j].Name
				}
			}
		}

		crtNode.ADist[0] = 0
		crtNode.pDist[0] = crtNode.Name

		for lvl := K-1 ; lvl >= 1 ; lvl-- {
			if crtNode.ADist[lvl] == crtNode.ADist[lvl-1] {
				crtNode.pDist[lvl-1] = crtNode.pDist[lvl]
			}
		}
	}
}

func computeCluster(nodeIdx int, nodes []Node, K int) {
	crtNode := &nodes[nodeIdx]

	for i := 0 ; i < len(nodes) ; i++ {
		if i != nodeIdx {
			targetLvl := crtNode.Level + 1

			if targetLvl > K {
				log.Error("lvl too large")
			}

			dist := euclidianDist(*crtNode, nodes[i])
			if targetLvl == K || dist < crtNode.ADist[targetLvl] {
				crtNode.cluster = append(crtNode.cluster, nodes[i].Name)
			}
		}
	}

	// add everyone who's closer to me than to anyone else at the next level

}


func alg(K int, nodes []Node) {
	for i := K-1 ; i >=0 ; i-- {
		for j := 0 ; j < len(nodes) ; j++ {
			if nodes[j].Level >= i {
				computeCluster(j, nodes, K)
			}
		}
	}
}

func main() {

	K := flag.Int("K", 3, "Number of levels.")
	N := flag.Int("N", 135, "Number of validators.")
	SpaceMax := flag.Int("SpaceMax", 150, "Coordinate space size.")

	flag.Parse()
	RndSrc = rand.New(rand.NewSource(time.Now().UnixNano()))
	nodes := genNodes(*N, *SpaceMax, *K)
	genLevels(*N, *K, nodes)

	computeADist(nodes, *K)

	alg(*K, nodes)

	for i := 0 ; i < *N ; i++ {
		//log.Lvl1(nodes[i].X, nodes[i].Y, nodes[i].Level, nodes[i].ADist, nodes[i].pDist)
		log.Lvl1(i, "<",nodes[i].X,",", nodes[i].Y, ">", "lvl=", nodes[i].Level, "cluster size=", len(nodes[i].cluster), nodes[i].cluster)
	}
	
}

