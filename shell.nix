{ pkgs ? (import <nixpkgs> {}).pkgs }:
with pkgs.python39Packages;

pkgs.mkShell{
  # venvDir = "./.venvDir";
  buildInputs = [
	# poetry env
	pkgs.python39
	pkgs.poetry
	numpy
	networkx
	pkgs.libspatialindex
	dateutil
	pandas
	Rtree

	# jupyter nb
	jupyter
	ipykernel
	ipywidgets
	
	];
}

